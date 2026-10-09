#!/usr/bin/env python
"""解析实验日志 → 生成 experiment_summary.xlsx（两个 sheet：汇总 + 每25epoch曲线）"""
import os
import re
import zipfile
import datetime

LOGS = [
    ('数据增强+bbox(4-shot)', 'logs/trn_btcv_elcseg_bbox_aug2.log/log.txt'),
    ('无增强+新bbox(4-shot)', 'logs/trn_btcv_elcseg_newbbox2.log/log.txt'),
    ('只用bbox(无增强)', 'logs/trn_btcv_elcseg_bboxonly.log/log.txt'),
    ('老bbox逻辑(4-shot)', 'logs/trn_btcv_elcseg_bbox.log/log.txt'),
    ('One-shot+注意力引导', 'logs/trn_btcv_elcseg_attnguide2_shot1.log/log.txt'),
]

ROW_RE = re.compile(r"\*\*\* (Training|Validation).*Avg L: [0-9.]+.*mIoU:\s+([0-9.]+)")
BEST_RE = re.compile(r"Model saved @(\d+) w/ val. mIoU:\s+([0-9]+(?:\.[0-9]+)?)")


def parse_log(path):
    trn, val, best = {}, {}, None
    epoch = 0
    with open(path) as f:
        for line in f:
            m = ROW_RE.search(line)
            if m:
                if m.group(1) == 'Training':
                    trn[epoch] = float(m.group(2))
                else:
                    val[epoch] = float(m.group(2))
                    epoch += 1
            b = BEST_RE.search(line)
            if b:
                best = (int(b.group(1)), float(b.group(2)))
    return trn, val, best


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def cell(col, row, value, numeric=False):
    r = f"{col}{row}"
    if numeric:
        return f'<c r="{r}"><v>{value}</v></c>'
    return f'<c r="{r}" t="inlineStr"><is><t>{esc(value)}</t></is></c>'


def make_sheet(name, rows):
    """rows: list of list（首行为表头）"""
    xml = [f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for i, row in enumerate(rows, start=1):
        xml.append(f'<row r="{i}">')
        for j, v in enumerate(row):
            col = chr(ord('A') + j)
            xml.append(cell(col, i, v, numeric=isinstance(v, float)))
        xml.append('</row>')
    xml.append('</sheetData></worksheet>')
    return ''.join(xml)


def build_xlsx(sheets, out_path):
    n = len(sheets)
    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                     '<Default Extension="xml" ContentType="application/xml"/>',
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(1, n + 1):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append('</Types>')

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>',
            '</Relationships>']

    sheets_xml = []
    wb_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(1, n + 1):
        wb_rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
    wb_rels.append('</Relationships>')

    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    for i, (name, _) in enumerate(sheets, start=1):
        wb.append(f'<sheet name="{esc(name)}" sheetId="{i}" r:id="rId{i}"/>')
    wb.append('</sheets></workbook>')

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ''.join(content_types))
        z.writestr('_rels/.rels', ''.join(rels))
        z.writestr('xl/workbook.xml', ''.join(wb))
        z.writestr('xl/_rels/workbook.xml.rels', ''.join(wb_rels))
        for i, (_, rows) in enumerate(sheets, start=1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', make_sheet(f'sheet{i}', rows))


def main():
    summary = [['实验配置', '训练 mIoU（末epoch）', '验证 mIoU（best）', 'gap', 'best epoch', '备注']]
    curves = [['epoch']]
    all_curves = {}

    for name, path in LOGS:
        trn, val, best = parse_log(path)
        max_e = max(val.keys()) if val else 0
        last_trn = trn.get(max_e, float('nan'))
        best_val = best[1] if best else float('nan')
        best_ep = best[0] if best else None
        summary.append([name, round(last_trn, 2), round(best_val, 2),
                        round(last_trn - best_val, 2), best_ep,
                        f'{max_e} epochs'])
        all_curves[(name + '-Trn')] = trn
        all_curves[(name + '-Val')] = val
        curves[0].extend([f'{name}-Trn', f'{name}-Val'])

    max_epochs = max(max(v.keys()) for v in all_curves.values() if v)
    for e in range(0, max_epochs + 1, 25):
        row = [e]
        for name, _ in LOGS:
            row.append(all_curves[(name + '-Trn')].get(e))
            row.append(all_curves[(name + '-Val')].get(e))
        curves.append(row)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_summary.xlsx')
    build_xlsx([('汇总', summary), ('每25epoch曲线', curves)], out)
    print(f'已生成: {out}')
    print('汇总表:')
    for r in summary:
        print('  ' + ' | '.join(str(x) for x in r))


if __name__ == '__main__':
    main()
