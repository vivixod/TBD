"""调试可视化: BTCV query=ABD_008_104, support=ABD_008_108.

加载 logs/trn_btcv_elcseg_shot4.log/best_model.pt (仅 VRP_encoder, 无 SAM 权重),
跑完整 VRP_encoder 前向流程, 抓取并可视化:
  A            = eicl_interaction 内 softmax(Q K^T / τ)     [1, HW, HW]
  pseudo_mask  = A @ M                                      [1, 1, 32, 32]
  attended_supp= A @ K                                      [1, 256, 32, 32]
  query_feat_1 = merge_1([q_down, attended, pseudo])        [1, 256, 32, 32]
  supp_feat_1  = merge_1([s_down, s_down*mask, mask])       [1, 256, 32, 32]

所有图与 npy 保存到 vis_eicl_debug/
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model.VRP_encoder import VRP_encoder

BASE = '/vcl/2025zhutaotao/BTCV'
CKPT = '/vcl/2025zhutaotao/VRP-SAM/logs/trn_btcv_elcseg_shot4.log/best_model.pt'
OUT = '/vcl/2025zhutaotao/VRP-SAM/vis_eicl_debug'
QUERY_ID = 'ABD_008_104'
SUPPORT_ID = 'ABD_008_108'

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

LABELS = {
    0: 'background', 1: 'spleen', 2: 'kidney_right', 3: 'kidney_left',
    4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
    9: 'inferior_vena_cava', 10: 'portal_vein_and_splenic_vein',
    11: 'pancreas', 12: 'adrenal_gland_right', 13: 'adrenal_gland_left',
}


def denorm(t):
    """[1,3,H,W] 归一化 tensor -> [H,W,3] uint8"""
    img = t[0].cpu().clone()
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(IMG_STD).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def load_pair(slice_id, transform):
    img = Image.open(os.path.join(BASE, 'image', 'x', f'{slice_id}.png')).convert('RGB')
    mask = np.array(Image.open(os.path.join(BASE, 'mask', f'x___{slice_id}.png')))
    t = transform(img)                      # [3,512,512]
    return t.unsqueeze(0), mask             # [1,3,512,512], [512,512]


def draw_mask_legend(ax, present_classes, cmap_use, n_all):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=cmap_use(c / (n_all - 1)),
                              label=f'{c}: {LABELS[c]}') for c in present_classes]
    ax.legend(handles=handles, loc='upper right', fontsize=6,
              bbox_to_anchor=(1.25, 1.0), title='Classes')


def feat_summary_plot(feat, title, save_path):
    """feat: [1,C,h,w] -> 保存 mean/std 与 4 个示例通道图"""
    x = feat[0].numpy()                       # [C,h,w]
    C = x.shape[0]
    chans = [0, max(1, C // 4), C // 2, C - 1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plots = [('channel mean', x.mean(0)),
             ('channel std', x.std(0)),
             (f'ch {chans[0]}', x[chans[0]]),
             (f'ch {chans[1]}', x[chans[1]]),
             (f'ch {chans[2]}', x[chans[2]]),
             (f'ch {chans[3]}', x[chans[3]])]
    for ax, (name, m) in zip(axes.flat, plots):
        im = ax.imshow(m, cmap='viridis')
        ax.set_title(name, fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cuda')
    print('device:', dev)

    # ---- 1. 构建模型并加载权重 (ckpt 为 DDP 扁平 state_dict, 去 module. 前缀) ----
    args = argparse.Namespace(num_query=50)
    model = VRP_encoder(args, backbone='resnet50', use_original_imgsize=False)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('load_state_dict -> missing:', len(missing), 'unexpected:', len(unexpected))
    if missing:
        print('  missing keys:', missing)
    if unexpected:
        print('  unexpected keys:', unexpected)
    model = model.to(dev)
    model.eval()

    # ---- 2. 数据准备 ----
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    q_img, q_mask = load_pair(QUERY_ID, transform)       # [1,3,512,512], [512,512]
    s_img, s_mask = load_pair(SUPPORT_ID, transform)

    q_cls = sorted(set(np.unique(q_mask)) - {0})
    s_cls = sorted(set(np.unique(s_mask)) - {0})
    common = sorted(set(q_cls) & set(s_cls))
    print('query classes:', {c: LABELS[c] for c in q_cls})
    print('support classes:', {c: LABELS[c] for c in s_cls})
    print('common classes:', {c: LABELS[c] for c in common})
    target = max(common, key=lambda c: int((s_mask == c).sum()))
    print(f'target organ: class {target} = {LABELS[target]}')
    support_mask_bin = (torch.from_numpy((s_mask == target).astype(np.float32))
                        .unsqueeze(0).unsqueeze(0))       # [1,1,512,512]

    # ---- 3. Hook 抓中间结果 ----
    cap = {'eicl': None, 'merge': []}

    def hook_eicl(m, inp, out):
        cap['eicl'] = [o.detach().cpu() for o in out]

    def hook_merge(m, inp, out):
        cap['merge'].append(out.detach().cpu())

    model.eicl_interaction.register_forward_hook(hook_eicl)
    model.merge_1.register_forward_hook(hook_merge)

    q_img_cuda = q_img.to(dev)
    s_img_cuda = s_img.to(dev)
    support_mask_cuda = support_mask_bin.to(dev)

    # ---- 4. 完整前向 (condition=mask, 与训练一致) ----
    with torch.no_grad():
        protos, bbox_token, pseudo_ret = model('mask', q_img_cuda, s_img_cuda,
                                               support_mask_cuda, training=False)
    A = model.eicl_interaction.last_A.cpu()                # [1, HW, HW]
    pseudo_mask, attended_supp = cap['eicl']               # [1,1,32,32], [1,256,32,32]
    query_feat_1 = cap['merge'][0]                          # [1,256,32,32]
    supp_feat_1 = cap['merge'][1]                           # [1,256,32,32]

    print('\n=== captured tensors ===')
    print('A           :', tuple(A.shape), 'dtype', A.dtype,
          'max %.4f mean %.4f' % (A.max(), A.mean()))
    print('pseudo_mask :', tuple(pseudo_mask.shape), 'range [%.3f, %.3f]' % (pseudo_mask.min(), pseudo_mask.max()))
    print('attended    :', tuple(attended_supp.shape))
    print('query_feat_1:', tuple(query_feat_1.shape))
    print('supp_feat_1 :', tuple(supp_feat_1.shape))
    print('protos      :', tuple(protos.shape))
    print('bbox_token  :', tuple(bbox_token.shape),
          'range [%.3f, %.3f]' % (bbox_token.min(), bbox_token.max()))

    # ---- 5. 保存 npy ----
    np.save(os.path.join(OUT, 'A.npy'), A.numpy())
    np.save(os.path.join(OUT, 'pseudo_mask.npy'), pseudo_mask.numpy())
    np.save(os.path.join(OUT, 'attended_supp.npy'), attended_supp.numpy())
    np.save(os.path.join(OUT, 'query_feat_1.npy'), query_feat_1.numpy())
    np.save(os.path.join(OUT, 'supp_feat_1.npy'), supp_feat_1.numpy())

    # ---- 6. 可视化 ----
    _, _, H, W = pseudo_mask.shape      # 实际空间尺寸 (如 64x64)
    print('spatial size of feature maps:', H, W)

    # 6.0 输入参考图
    n_all = len(LABELS)
    cmap = plt.cm.get_cmap('tab20', n_all)
    cmap_use = ListedColormap([cmap(i)[:3] for i in range(n_all)])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(denorm(q_img)); axes[0, 0].set_title('query image (ABD_008_104)'); axes[0, 0].axis('off')
    axes[0, 1].imshow(q_mask, cmap=cmap_use, vmin=0, vmax=n_all - 1, interpolation='nearest')
    axes[0, 1].set_title('query mask (multi-class)'); axes[0, 1].axis('off')
    axes[1, 0].imshow(denorm(s_img)); axes[1, 0].set_title('support image (ABD_008_108)'); axes[1, 0].axis('off')
    axes[1, 1].imshow(support_mask_bin[0, 0], cmap='gray')
    axes[1, 1].set_title(f'support binary mask (class {target} = {LABELS[target]})'); axes[1, 1].axis('off')
    fig.suptitle('Inputs')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '0_inputs.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6.1 A 全矩阵热力图 (对数色标, 论文配图)
    from matplotlib.colors import LogNorm
    A_np = A[0].numpy()
    HW = A_np.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    vmin = max(float(A_np[A_np > 0].min()), 1e-6)
    A_plot = np.clip(A_np, vmin, None)
    im = ax.imshow(A_plot, cmap='magma', norm=LogNorm(vmin=vmin, vmax=A_np.max()),
                   aspect='equal', interpolation='nearest')
    # 每 64 个 patch 一条细网格线 (对应 64x64 空间网格)
    for g in range(0, HW + 1, 64):
        ax.axhline(g, color='white', lw=0.4, alpha=0.35)
        ax.axvline(g, color='white', lw=0.4, alpha=0.35)
    ax.set_xticks(np.arange(0, HW + 1, 1024))
    ax.set_yticks(np.arange(0, HW + 1, 1024))
    ax.set_xlabel('support patch index', fontsize=12)
    ax.set_ylabel('query patch index', fontsize=12)
    ax.set_title(f'Attention matrix  A = softmax(Q K\u1d40 / \u03c4)  [{HW}\u00d7{HW}]',
                 fontsize=13)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('attention weight (log scale)', fontsize=11)
    ax.text(0.02, 0.03, 'diagonal = spatially aligned patches',
            transform=ax.transAxes, color='white', fontsize=9, va='bottom')
    fig.savefig(os.path.join(OUT, '1_A_full.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 6.2 选 6 个 query patch, 看它们的注意力在 support 上的分布
    q_vis = denorm(q_img)                       # [512,512,3]
    cell = 512 // H
    rows = [max(0, r * H // 32) for r in [4, 8, 16, 16, 24, 28]]
    cols = [max(0, c * W // 32) for c in [8, 24, 4, 28, 16, 24]]
    fig, axes = plt.subplots(6, 2, figsize=(8, 22))
    for i, (r, c) in enumerate(zip(rows, cols)):
        idx = r * W + c
        att = A[0, idx].reshape(H, W).numpy()
        # 左: query 图上标记该 patch 位置 (grid cell = 512/H px)
        mark = q_vis.copy()
        y0, x0 = r * cell, c * cell
        mark[y0:y0 + cell, x0:x0 + cell] = [255, 0, 0]
        axes[i, 0].imshow(mark); axes[i, 0].axis('off')
        axes[i, 0].set_title(f'query patch ({r},{c}) idx={idx}')
        # 右: 注意力热力图 (support 空间)
        im = axes[i, 1].imshow(att, cmap='hot', vmin=0, vmax=att.max())
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'attention over support, max={att.max():.3f}')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
    fig.suptitle('Patch-level attention: selected query patches -> support')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '2_A_query_patches.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6.3 pseudo_mask
    pm = pseudo_mask[0, 0].numpy()             # [H,W]
    pm_up = F.interpolate(torch.from_numpy(pm).unsqueeze(0).unsqueeze(0),
                          size=(512, 512), mode='bilinear',
                          align_corners=True)[0, 0].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(q_vis); axes[0].set_title('query image'); axes[0].axis('off')
    im1 = axes[1].imshow(pm_up, cmap='jet'); axes[1].set_title('pseudo_mask'); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    overlay = q_vis.copy().astype(np.float32) / 255.0
    alpha = 0.6
    red = np.array([1.0, 0.0, 0.0])
    overlay = overlay * (1 - alpha * pm_up[..., None]) + red * (alpha * pm_up[..., None])
    axes[2].imshow(overlay); axes[2].set_title('pseudo_mask overlay'); axes[2].axis('off')
    fig.suptitle('pseudo_mask = A @ M  (min-max normalized)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '3_pseudo_mask.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6.4~6.6 特征图
    feat_summary_plot(attended_supp, f'attended_supp = A @ K  {tuple(attended_supp.shape)}',
                      os.path.join(OUT, '4_attended_supp.png'))
    feat_summary_plot(query_feat_1, f'query_feat_1 = merge_1([q_down, attended, pseudo]) {tuple(query_feat_1.shape)}',
                      os.path.join(OUT, '5_query_feat_1.png'))
    feat_summary_plot(supp_feat_1, f'supp_feat_1 = merge_1([s_down, s_down*mask, mask]) {tuple(supp_feat_1.shape)}',
                      os.path.join(OUT, '6_supp_feat_1.png'))

    # ---- 7. 信息日志 ----
    stats = [
        f'query  : {QUERY_ID}  classes={ {c: LABELS[c] for c in q_cls} }',
        f'support: {SUPPORT_ID}  classes={ {c: LABELS[c] for c in s_cls} }',
        f'target : class {target} = {LABELS[target]}',
        '',
        'captured tensors:',
        f'  A            {tuple(A.shape)}  max={A.max():.4f} mean={A.mean():.4f}',
        f'  pseudo_mask  {tuple(pseudo_mask.shape)}  range=[{pseudo_mask.min():.3f},{pseudo_mask.max():.3f}]',
        f'  attended_supp {tuple(attended_supp.shape)}  mean_abs={attended_supp.abs().mean():.4f}',
        f'  query_feat_1 {tuple(query_feat_1.shape)}',
        f'  supp_feat_1  {tuple(supp_feat_1.shape)}',
        f'  protos       {tuple(protos.shape)}',
        f'  bbox_token   {tuple(bbox_token.shape)}',
        '',
        'checkpoint vs code (权重差异):',
        f'  missing ({len(missing)}): {missing}',
        f'  unexpected ({len(unexpected)}): {unexpected}',
    ]
    with open(os.path.join(OUT, 'stats.txt'), 'w') as f:
        f.write('\n'.join(stats) + '\n')
    print('\n'.join(stats))
    print('\nall saved to:', OUT)


if __name__ == '__main__':
    main()
