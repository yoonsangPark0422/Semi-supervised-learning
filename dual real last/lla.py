"""LLA core, independently implemented from Lee et al., ICCV 2025, Eqs. 9-15.

This is an integration into Dual, not the authors' official implementation.
No EFCC. Initialization, optimizer settings and stop-gradient choices are
explicitly documented in README.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


def soft_cross_entropy(logits, targets):
    """Per-sample soft CE. Targets must never train themselves."""
    return -(targets.detach() * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def class_average(logits, targets, eps=1e-8):
    """Eqs. 11/12 and 14/15: mean over ALL K classes, not present classes.

    For unlabeled data the denominator is soft target mass, not argmax counts.
    Empty labeled classes contribute zero (division uses +eps).
    """
    targets = targets.detach().to(logits.dtype)
    numerator = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=0)
    denominator = targets.sum(dim=0) + eps
    return (numerator / denominator).mean()


class LearnableLogitAdjustment(nn.Module):
    def __init__(self, num_classes, eps=1e-8):
        super().__init__()
        if num_classes < 2 or eps <= 0:
            raise ValueError('Expected >=2 classes and positive epsilon.')
        # Uniform softmax(lambda) starts with no relative logit adjustment.
        self.adjustment = nn.Parameter(torch.zeros(num_classes))
        self.eps = eps

    def log_prior(self):
        return F.log_softmax(self.adjustment, dim=0)

    def adjust(self, logits):
        return logits - self.log_prior()

    @torch.no_grad()
    def pseudo_labels(self, weak_logits):
        return F.softmax(self.adjust(weak_logits.detach()), dim=-1)

    def class_averaged_losses(self, labeled_logits, labels, strong_logits, soft_targets):
        # Lc must not propagate into either CNN or the pseudo-label branch.
        one_hot = F.one_hot(labels, num_classes=self.adjustment.numel()).to(labeled_logits.dtype)
        labeled = class_average(self.adjust(labeled_logits.detach()), one_hot, self.eps)
        unlabeled = class_average(self.adjust(strong_logits.detach()), soft_targets, self.eps)
        return labeled, unlabeled
