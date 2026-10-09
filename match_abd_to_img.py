#!/usr/bin/env python
"""快速版：只在 MASS 的 val+test 7 个 nii 里找 ABD 验证患者的对应（向量化 NCC）。"""
import glob
import os

import cv2
import numpy as np
import nibabel as nib
import PIL.Image

BTCV_2D = '/vcl/2025zhutaotao/BTCV/image/x'
RAWDATA = '/vcl/2025zhutaotao/VRP-SAM/RawData/Training/img'
ABD_PATIENTS = ['ABD_034', 'ABD_035', 'ABD_036', 'ABD_037', 'ABD_038', 'ABD_039', 'ABD_040']
MASS_VALTEST = ['img0032', 'img0006', 'img0038', 'img0022', 'img0034', 'img0026', 'img0031']


def load_nii_small(path, size=96):
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)   # [x, y, z]
    vol = np.transpose(vol, (2, 0, 1))                            # [z, x, y]
    vol = np.clip(vol, -200, 500)
    nz = vol.shape[0]
    s = 512 // size                      # 步长（512/96 ≈ 5）
    small = vol[:, 1:1 + size * s:s, 1:1 + size * s:s]   # [z, size, size]
    f = small.reshape(nz, -1)
    f = f - f.mean(1, keepdims=True)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f


def load_pngs_small(patient, size=96, max_n=10):
    files = sorted(glob.glob(os.path.join(BTCV_2D, f'{patient}_*.png')))[:max_n]
    arrs = []
    for fl in files:
        a = np.array(PIL.Image.open(fl).convert('L'), dtype=np.float32)
        arrs.append(a)
    stack = np.stack(arrs)              # [M, 512, 512]
    s = 512 // size
    f = stack[:, 1:1 + size * s:s, 1:1 + size * s:s].reshape(len(arrs), -1)
    f = f - f.mean(1, keepdims=True)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f


def main():
    for p in ABD_PATIENTS:
        q = load_pngs_small(p)
        scores = []
        for name in MASS_VALTEST:
            path = os.path.join(RAWDATA, f'{name}.nii.gz')
            if not os.path.exists(path):
                print(f'{name} 不存在'); continue
            f = load_nii_small(path)
            sim = f @ q.T
            best = sim.max()
            scores.append((name, float(best), int(sim.argmax() // sim.shape[1])))
        scores.sort(key=lambda x: -x[1])
        print(f'{p}: ' + ' | '.join(f'{n}({s:.3f},z={z})' for n, s, z in scores))


if __name__ == '__main__':
    main()
