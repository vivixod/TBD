"""Build support spatial masks without changing query attention or retrieval."""
def support_background_mask(support_mask, features):
    """Return [B, HW] bool padding mask: True excludes a background key.

    Accept binary/occupancy masks [B, H, W] or [B, 1, H, W] already aligned
    with the support feature grid. Resizing is handled once by VRP_encoder.
    """
    if support_mask is None:
        raise ValueError('Support attention masking requires a support mask')
    if support_mask.ndim == 4 and support_mask.shape[1] == 1:
        support_mask = support_mask[:, 0]
    if (support_mask.ndim != 3
            or support_mask.shape[0] != features.shape[0]
            or support_mask.shape[-2:] != features.shape[-2:]):
        raise ValueError('Support mask must match the support feature batch and spatial grid')
    return ~(support_mask.flatten(1) > 0)
