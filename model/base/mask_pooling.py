"""Support mask resizing policies, including dataset-aware joint batches."""
import torch
import torch.nn.functional as F


def repeat_dataset_names(dataset_names, batch_size, repeats=1):
    """Match query.repeat_interleave(shot) / support.flatten(0, 1) order."""
    if dataset_names is None:
        return None
    names = [dataset_names] * batch_size if isinstance(dataset_names, str) else list(dataset_names)
    if len(names) != batch_size:
        raise ValueError('Expected one dataset name per query in the batch')
    return [name for name in names for _ in range(repeats)]


def resize_support_mask(mask, output_size, policy='max', dataset_names=None):
    """Resize [B, 1, H, W] support masks using max, nearest, or adam-only.

    adam-only uses max pooling on ADAM samples and nearest on all other samples.
    Dataset identity must come from the episode, never its class ID or filename.
    """
    if policy == 'max':
        return F.adaptive_max_pool2d(mask, output_size)
    if policy == 'nearest':
        return F.interpolate(mask, size=output_size, mode='nearest')
    if policy != 'adam-only':
        raise ValueError(f'Unknown support mask pooling policy: {policy}')
    names = repeat_dataset_names(dataset_names, mask.shape[0])
    if names is None or any(name is None or name == 'joint' for name in names):
        raise ValueError('adam-only pooling requires each episode\'s dataset name')
    use_max = [name == 'adam' for name in names]
    if all(use_max):
        return F.adaptive_max_pool2d(mask, output_size)
    if not any(use_max):
        return F.interpolate(mask, size=output_size, mode='nearest')
    selector = torch.tensor(use_max, dtype=torch.bool, device=mask.device)[:, None, None, None]
    return torch.where(selector, F.adaptive_max_pool2d(mask, output_size),
                       F.interpolate(mask, size=output_size, mode='nearest'))
