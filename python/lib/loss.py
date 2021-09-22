import torch


def cross_entropy_masked(logits, target, mask):
    assert logits.shape == target.shape, (logits.shape, target.shape)
    assert mask is None or logits.shape == mask.shape, (logits.shape, mask.shape)

    # flatten all except the batch dimension
    batch_size = len(logits)
    logits = logits.view(batch_size, -1)
    target = target.view(batch_size, -1)
    mask = mask.view(batch_size, -1) if mask is not None else None

    # ignore empty masks
    if mask is not None:
        mask_empty = mask.sum(dim=1) == 0
        logits = logits[~mask_empty, :]
        target = target[~mask_empty, :]
        mask = mask[~mask_empty, :]

    # the mechanism for ignoring masked values:
    #   log converts 0->-inf, 1->0
    #   log_softmax converts -inf->nan
    #   nansum then ignores these propagated nan values
    mask_log = mask.log() if mask is not None else 0
    log = torch.log_softmax(logits + mask_log, dim=1)
    loss = -(target * log).nansum(dim=1)

    assert loss.isfinite().all(), \
        "inf/nan values in loss, maybe the mask and target contain impossible combinations?"

    # average over batch dimension
    return loss.mean(dim=0)
