"""Explicit presets from the supplied runner; these are objectives, not four trunks."""
from copy import deepcopy

BASE = dict(window=10, tiles=4, gap=(1, 8), dim=64, width=64, head_width=64,
            levels=8, order=1.0, pair=0.5, reconstruct=1.0, lag=0.0,
            lag_classes=6, head="deepsets", neuron_dropout=0.1, gain_jitter=0.1,
            order_augment_scale=1.0, time_roll=0, hard_fraction=1.0, reset_every=0)
PRESETS = {
    "proposed": {},
    "order_default": {},
    "order_lag": dict(lag=1.0),
    "order_time_axis": dict(order=0.0, pair=0.0, lag=1.0, head="linear", lag_classes=8),
    "order_full": dict(window=6, gap=(3, 5), order_augment_scale=4.0,
                       time_roll=2, lag=1.0, lag_classes=8,
                       hard_fraction=0.25, reset_every=400),
    "order_gapped": dict(window=6, gap=(3, 5)),
    "reconstruct_only": dict(order=0.0, pair=0.0),
    "anchor_default": dict(order=0.0, pair=0.0),
    "anchor_full": dict(window=6, gap=(3, 5), order=0.0, pair=0.0),
}
DEFAULT_ARMS = ("proposed", "order_lag", "order_time_axis", "order_full")


def preset(name, **overrides):
    if name not in PRESETS:
        raise ValueError(f"Unknown arm {name!r}; choose from {list(PRESETS)}")
    return dict(deepcopy(BASE), **PRESETS[name]) | overrides
