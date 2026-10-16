"""Two CPU ranks exercising the real quality prompt model and two DDP graphs."""
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segment-anything'))
from model.VRP_encoder import VRP_encoder
from model.base import resnet
from model.base.support_fusion import fuse_support_logits, support_quality_loss


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 1)
    def forward(self, prompts):
        return self.projection(prompts.mean(1)).reshape(1, 4, 1, 1).expand(-1, -1, 8, 8)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(321 + rank)
        factory = resnet.resnet50
        args = SimpleNamespace(num_query=50, support_fusion='quality', support_mask_pooling='max',
                               bbox_support_attn_mask=1, icl_support_attn_mask=1)
        with mock.patch('model.VRP_encoder.models.resnet50', side_effect=lambda **kw: factory(pretrained=False)):
            prompt = VRP_encoder(args, 'resnet50', False)
        for layer in [prompt.layer0, prompt.layer1, prompt.layer2, prompt.layer4]:
            layer.requires_grad_(False)
        prompt.train_mode()
        model = nn.parallel.DistributedDataParallel(prompt, find_unused_parameters=True)
        sam = nn.parallel.DistributedDataParallel(Decoder(), find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]
                                      + list(sam.parameters()), lr=1e-3)
        initial = prompt.support_quality_head.mlp[-1].weight.detach().clone()
        for epoch in [0, 4]:
            prompt.set_fusion_epoch(epoch)
            query = torch.randn(1, 3, 32, 32).repeat_interleave(4, 0)
            support = torch.randn(4, 3, 32, 32)
            mask = torch.zeros(4, 32, 32); mask[:, 1:3, 1:3] = 1
            tokens, boxes, _, scores = model('mask', query, support, mask, True, return_quality=True)
            logits = sam(torch.cat([tokens, boxes], 1))
            fused, _ = fuse_support_logits(logits, scores.reshape(1, 4), prompt.support_quality_head.mix)
            # Synthetic teacher guarantees a nonuniform label for this graph check.
            teacher_logits = torch.full_like(logits, -10); teacher_logits[:, 0] = 10
            quality, _, _ = support_quality_loss(scores.reshape(1, 4), teacher_logits, torch.ones(1, 8, 8))
            loss = fused.square().mean() + .1 * quality + .01 * boxes.square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            assert torch.isfinite(loss)
        weight = prompt.support_quality_head.mlp[-1].weight.detach()
        assert not torch.equal(initial, weight)
        copies = [torch.zeros_like(weight) for _ in range(2)]
        dist.all_gather(copies, weight)
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
        if rank == 0:
            print('PASS: two real prompt-model ranks, warmup/full fusion, gradients, synchronized quality weights', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)
