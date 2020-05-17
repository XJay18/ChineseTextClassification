import torch
import torch.nn.functional as F


def binary_focal_loss(input, target, alpha=0.25, gamma=2,
                      ignore_index=-100, reduction='mean'):
    weight = torch.tensor([alpha, 1 - alpha], device=input.device)
    probs = F.softmax(input, dim=-1)
    log_probs = torch.log(probs)
    logits = (1 - probs) ** gamma * log_probs

    return F.nll_loss(logits, target, weight=weight,
                      ignore_index=ignore_index, reduction=reduction)


LOSSES = {
    "CE": F.cross_entropy,
    "FocalCE": binary_focal_loss,
}


def fetch_loss(name="CE"):
    assert name in LOSSES, "The loss '%s' currently is not supported." % name
    print("Using %s loss." % name)
    return LOSSES[name]
