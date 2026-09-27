"""Primitives extracted verbatim from the user-supplied Jigsaw sources.
See SOURCES.json for hashes and symbols. No MobileJigsaw dependency.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
TRUNK_BLOCKS = ("residual", "separable")
TILE_NORMS = ("none", "mean", "global_mean", "zscore")


class _GradScale(torch.autograd.Function):
    """Straight-through forward, scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class _Residual(nn.Module):
    """Valid 3-tap residual block; consumes exactly two bins."""

    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(width, width, 3), nn.GELU(),
            nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Separable(nn.Module):
    """MobileNetV2-style inverted residual with a VALID depthwise 3-tap conv.

    1x1 expand -> GELU -> depthwise temporal conv -> GELU -> 1x1 linear project.
    Consumes exactly two bins, like _Residual, so trunk arithmetic is unchanged.
    """

    def __init__(self, width, dropout, expansion=4):
        super().__init__()
        hidden = width * expansion
        self.net = nn.Sequential(
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(width, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, groups=hidden), nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Trunk(nn.Module):
    """Valid-convolution stack whose receptive field is exactly window_size."""

    def __init__(self, channels, window_size, width, dropout, block="residual"):
        super().__init__()
        if block not in TRUNK_BLOCKS:
            raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
        first_kernel = 2 if window_size % 2 == 0 else 3
        blocks = (window_size - first_kernel - 2) // 2
        make = _Residual if block == "residual" else _Separable
        self.layers = nn.Sequential(
            nn.Conv1d(channels, width, first_kernel),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(), nn.GELU(),
            *[make(width, dropout) for _ in range(blocks)],
            nn.Conv1d(width, width, 3), nn.GELU(),
        )

    def forward(self, x):
        h = self.layers(x)
        if h.shape[-1] != 1:
            raise ValueError("Trunk expects exactly window_size input bins.")
        return h.squeeze(-1)


class _Encoder(nn.Module):
    def __init__(self, channels, window_size, width, output_dimension, dropout,
                 normalize, trunk_block):
        super().__init__()
        self.trunk = _Trunk(channels, window_size, width, dropout, trunk_block)
        self.project = nn.Linear(width, output_dimension)
        self.normalize = normalize

    def forward(self, x):
        z = self.project(self.trunk(x))
        return F.normalize(z, dim=-1) if self.normalize else z


class _OrderHead(nn.Module):
    """Permutation-equivariant DeepSets head. Two cross-entropy readouts.

    position : K-way logits per tile, "which time slot is this tile?"
    pair     : one scalar per tile; their difference is the before/after logit.
    """

    def __init__(self, dimension, hidden, n_tiles):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.position = nn.Linear(hidden, n_tiles)
        self.score = nn.Linear(hidden, 1)

    def forward(self, z):
        context = z.mean(dim=1, keepdim=True).expand_as(z)
        tokens = self.body(torch.cat((z, context), dim=-1))
        return self.position(tokens), self.score(tokens).squeeze(-1)


class _ReconstructHead(nn.Module):
    """Per-neuron, per-bin cross-entropy over the quantized tile the embedding came from.

    The anti-collapse anchor. The order task only needs a low-dimensional
    "where in time am I" code and will happily throw everything else away --
    measured as a participation ratio of 3.4 out of 64 and ridge R2 of 0.005.
    Forecasting one bin is too weak a constraint to stop that. Requiring the
    embedding to reproduce EVERY bin of its own window forces it to stay an
    information-preserving bottleneck, and the raw window is exactly what the
    decoding ceiling is computed from, so this term targets the measured gap
    rather than a proxy for it. Still nothing but cross-entropy.
    """

    def __init__(self, dimension, hidden, channels, window_size, levels):
        super().__init__()
        self.channels, self.window_size, self.levels = channels, window_size, levels
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels * window_size * levels))

    def forward(self, z):
        return self.net(z).reshape(-1, self.channels, self.window_size, self.levels)


def _tile_normalize(tiles, mode):
    if mode == "none":
        return tiles
    if mode == "mean":
        return tiles - tiles.mean(dim=3, keepdim=True)
    if mode == "global_mean":
        return tiles - tiles.mean(dim=(2, 3), keepdim=True)
    if mode == "zscore":
        return (tiles - tiles.mean(dim=3, keepdim=True)) / (tiles.std(dim=3, keepdim=True) + 1e-5)
    raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")


def _pair_targets(positions):
    """Upper-triangular before/after labels and the mask of valid pairs."""
    difference = positions[:, :, None] - positions[:, None, :]
    mask = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)
    return (difference < 0).to(torch.float32), mask


class _LinearOrderHead(nn.Module):
    """The lowest-capacity head that can express the task at all.

        position logits = W z_k + b          (per tile, no cross-tile context)
        pair logit(i,j) = w^T (z_i - z_j)    (one direction, exactly antisymmetric)

    Two reasons this is the interesting head and not a crippled one.

    It cannot memorize. A 64x4 matrix plus a 64-vector has 324 parameters
    against 3752 training spans, so there is no lookup table available; if the
    order loss goes down, the order information went into the EMBEDDING, which
    is the only thing we actually care about. The DeepSets head has ~12k
    parameters and drove its own loss to 0.0026 while the embedding learned
    nothing -- that is the failure this replaces.

    It states a strong, quotable property. `pair logit = w^T (z_i - z_j)` says
    time is a single LINEAR DIRECTION in embedding space; driving that BCE down
    means the encoder has laid the trajectory out along an axis, which is
    exactly the geometry a linear decoder wants. Permutation equivariance is
    automatic here because nothing mixes tiles.
    """

    def __init__(self, dimension, n_tiles):
        super().__init__()
        self.position = nn.Linear(dimension, n_tiles)
        self.score = nn.Linear(dimension, 1, bias=False)

    def forward(self, z):
        return self.position(z), self.score(z).squeeze(-1)


class _LagHead(nn.Module):
    """Classify the SIGNED, QUANTIZED bin offset between two tiles.

    The slot label is worth log K = 1.39 nats and is blind to distance: two
    tiles 11 bins apart and two tiles 51 bins apart carry the same target. The
    lag label is worth log(lag_classes) nats and is not. It also cannot be
    solved by any monotone statistic on its own, because getting the class right
    needs the MAGNITUDE of the separation, not just its sign -- so the level
    drift that hands over the permutation for free does not hand this over.

    Reads `z_i - z_j`, so it is a pure relation and adds nothing per-tile.
    Still one `F.cross_entropy`.
    """

    def __init__(self, dimension, hidden, classes, *, linear=False):
        super().__init__()
        self.classes = classes
        self.net = nn.Linear(dimension, classes) if linear else nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, classes))

    def forward(self, difference):
        return self.net(difference)


def lag_edges(window_size, n_tiles, gap_low, gap_high, classes, device):
    """Geometric magnitude bands, mirrored around zero. Even class count.

    Reachable |offset| runs from window_size+gap_low (neighbours, tightest gap)
    to (K-1)*window_size + (K-1)*gap_high (the outermost pair, widest gaps).
    Geometric rather than linear because the distribution of |offset| is heavily
    weighted toward the small end and equal-width bands would leave the top
    classes nearly empty -- the same argument that made the forecast targets
    per-neuron quantiles instead of equal-width bins.
    """
    bands = classes // 2
    low = float(window_size + gap_low)
    high = float((n_tiles - 1) * (window_size + gap_high))
    inner = torch.logspace(math.log10(low), math.log10(max(high, low + 1.0)),
                           bands + 1, device=device)[1:-1]
    return inner


def lag_targets(tile_starts, labels, edges, classes):
    """(rows,) class index for every upper-triangular pair of presented slots.

    `tile_starts` is (B, K) in CHRONOLOGICAL order; `labels[b, s]` is the
    chronological index of whichever tile is sitting in presented slot s, which
    is exactly what `_shuffle_tiles` returns. Gathering by `labels` puts the
    starts back in PRESENTED order so the target lines up with the embeddings.

    Orientation follows `_pair_targets` in jigsaw_net.py: entry [i, j] is about
    x_i - x_j, so the feature fed to the head must be `z_i - z_j` with the same
    indexing. Getting this backwards would train the head on mirrored labels and
    still look like it was learning, so it is worth stating twice.
    """
    presented = torch.gather(tile_starts, 1, labels)
    delta = presented[:, :, None] - presented[:, None, :]            # [i, j] = t_i - t_j
    mask = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
    magnitude = torch.bucketize(delta.abs()[mask].to(torch.float32), edges)
    bands = classes // 2
    later = (delta[mask] > 0)                                        # slot i came AFTER slot j
    return torch.where(later, bands + magnitude,
                       bands - 1 - magnitude).clamp(0, classes - 1), mask
