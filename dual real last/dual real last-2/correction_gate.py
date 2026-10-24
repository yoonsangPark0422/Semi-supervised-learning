"""Narrow correction gate: no ground truth enters target selection."""


def correction_gate(major, minor_raw, minor_adjusted, major_is_head,
                    minor_adjusted_is_tail, major_to_minor_sent):
    return (major_to_minor_sent.bool() & major_is_head.bool()
            & minor_adjusted_is_tail.bool()
            & major.eq(minor_raw) & major.ne(minor_adjusted))


def suppress_conflicting_self(self_mask, gate):
    return self_mask * (~gate).to(self_mask.dtype)


def reference_masked_mean(losses, active_mask, reference_mask):
    # Keep every untouched sample's original coefficient.
    return (losses * active_mask).sum() / reference_mask.sum().clamp_min(1.0)
