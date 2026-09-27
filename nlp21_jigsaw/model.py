"""Neurons -> shared Jigsaw encoder -> embedding windows -> packed GRU -> CTC."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .config import preset
from .primitives import (_Encoder, _OrderHead, _ReconstructHead, _LinearOrderHead,
                         _LagHead, _tile_normalize, _pair_targets, lag_edges, lag_targets)


def time_mask(x, lengths):
    return (torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None])[..., None]


class JigsawCTC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = dict(args)
        self.spec = preset(args["jigsaw_arm"], dim=args.get("jigsaw_dim", 64),
                           width=args.get("jigsaw_width", 64), head_width=args.get("head_width", 64))
        # The requested experiment adds only Gaussian corruption. The source's
        # extra gain/dropout/roll augmentations are available as an explicit opt-in.
        if not args.get("source_augment", False):
            self.spec.update(neuron_dropout=0.0, gain_jitter=0.0, time_roll=0)
        s = self.spec
        self.channels = args.get("nInputFeatures", 192)
        self.kernel, self.stride = args["kernel"], args["stride"]
        self.encoder = _Encoder(self.channels, s["window"], s["width"], s["dim"],
                                0.0, False, "residual")
        self.order_head = (_LinearOrderHead(s["dim"], s["tiles"]) if s["head"] == "linear"
                           else _OrderHead(s["dim"], s["head_width"], s["tiles"]))
        self.lag_head = _LagHead(s["dim"], s["head_width"], s["lag_classes"], linear=s["head"] == "linear")
        self.reconstruct_head = _ReconstructHead(s["dim"], s["head_width"], self.channels,
                                                s["window"], s["levels"])
        self.register_buffer("quantile_edges", torch.zeros(s["levels"] - 1, self.channels))
        self.register_buffer("quantiles_ready", torch.tensor(False))
        self.register_buffer("lag_boundaries", lag_edges(s["window"], s["tiles"], *s["gap"], s["lag_classes"], "cpu"))
        grid = torch.arange(20, dtype=torch.float32) - 9.5
        g = torch.exp(-0.5 * (grid / 2.0).square()); g /= g.sum()
        self.register_buffer("gaussian_kernel", g[None, None].repeat(self.channels, 1, 1))
        self.smoothing = not args.get("no_gauss", False)
        rnn = nn.GRU if args.get("gru", True) else nn.LSTM
        self.rnn = rnn(s["dim"] * self.kernel, args["hidden"], args["layers"],
                       batch_first=True, bidirectional=args.get("bidir", False),
                       dropout=args["dropout"] if args["layers"] > 1 else 0.0)
        self.classifier = nn.Linear(args["hidden"] * (2 if args.get("bidir", False) else 1), 32)
        self.encoder_frozen = False

    def ssl_parameters(self):
        for module in (self.encoder, self.order_head, self.lag_head, self.reconstruct_head):
            yield from module.parameters()

    def decoder_parameters(self):
        yield from self.rnn.parameters()
        yield from self.classifier.parameters()

    def freeze_encoder(self):
        self.encoder_frozen = True
        for p in self.ssl_parameters():
            p.requires_grad_(False)
            p.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            for module in (self.encoder, self.order_head, self.lag_head, self.reconstruct_head):
                module.eval()
        return self

    def preprocess(self, x, lengths):
        # Zero outside each trial BEFORE smoothing: predictions must not depend
        # on the length/content of the other trials in this padded batch.
        mask = time_mask(x, lengths)
        x = x * mask
        if self.smoothing:
            x = F.conv1d(F.pad(x.transpose(1, 2), (9, 10)), self.gaussian_kernel,
                         groups=self.channels).transpose(1, 2)
        return x * mask

    def encode_preprocessed(self, x, lengths):
        # Fully convolutional evaluation is algebraically the same as applying
        # the supplied _Encoder to every natural, edge-padded W-bin window.
        w = self.spec["window"]
        index = torch.arange(x.shape[1] + w - 1, device=x.device)[None] - w // 2
        index = index.expand(x.shape[0], -1).clamp_min(0)
        index = torch.minimum(index, lengths.to(x.device)[:, None] - 1)
        padded = x.gather(1, index[..., None].expand(-1, -1, x.shape[2]))
        h = self.encoder.trunk.layers(padded.transpose(1, 2)).transpose(1, 2)
        return self.encoder.project(h) * time_mask(x, lengths)

    def encode(self, x, lengths):
        return self.encode_preprocessed(self.preprocess(x, lengths), lengths)

    def output_lengths(self, lengths):
        return torch.div(lengths - self.kernel, self.stride, rounding_mode="floor") + 1

    def decode(self, z, lengths):
        out_lengths = self.output_lengths(lengths)
        if (out_lengths <= 0).any():
            raise ValueError("A trial is shorter than --kernel; lower kernel or inspect the data.")
        # Match the source Unfolder's channel-major feature order; include the
        # final valid window (+1), which the old length calculation omitted.
        windows = z.unfold(1, self.kernel, self.stride).flatten(2).contiguous()
        packed = pack_padded_sequence(windows, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(packed, batch_first=True, total_length=windows.shape[1])
        return self.classifier(h), out_lengths

    def forward(self, x, lengths):
        return self.decode(self.encode(x, lengths), lengths)

    @torch.no_grad()
    def set_quantiles(self, values):
        q = torch.linspace(0, 1, self.spec["levels"] + 1, device=values.device)[1:-1]
        self.quantile_edges.copy_(torch.quantile(values.float(), q, dim=0))
        self.quantiles_ready.fill_(True)

    def bucketize(self, values):
        result = torch.zeros_like(values, dtype=torch.long)
        for edge in self.quantile_edges:
            result += (values > edge).long()
        return result

    def sample_plan(self, x, lengths, count):
        """Draw within-trial spans, with clean reconstruction targets."""
        s = self.spec; w, k = s["window"], s["tiles"]
        max_span = k * w + (k - 1) * s["gap"][1]
        eligible = (lengths >= max_span).nonzero().flatten().to(x.device)
        if not len(eligible):
            raise ValueError(f"No trial in this batch has {max_span} bins for {self.args['jigsaw_arm']}.")
        trial = eligible[torch.randint(len(eligible), (count,), device=x.device)]
        start = (torch.rand(count, device=x.device) * (lengths.to(x.device)[trial] - max_span + 1)).long()
        gaps = torch.randint(s["gap"][0], s["gap"][1] + 1, (count, k - 1), device=x.device)
        relative = torch.cat((torch.zeros(count, 1, device=x.device, dtype=torch.long),
                              (w + gaps).cumsum(1)), dim=1)
        starts = start[:, None] + relative
        times = starts[..., None] + torch.arange(w, device=x.device)
        def gain(scale):
            keep = torch.rand(count, k, self.channels, 1, device=x.device) >= min(0.95, s["neuron_dropout"] * scale)
            factor = 1 + s["gain_jitter"] * scale * (2 * torch.rand(count, k, self.channels, 1, device=x.device) - 1)
            return keep * factor
        perm = torch.rand(count, k, device=x.device).argsort(1)
        amount = min(s["time_roll"], w - 1)
        shifts = torch.randint(-amount, amount + 1, (count, k), device=x.device)
        roll = (torch.arange(w, device=x.device)[None, None] - shifts[..., None]) % w
        clean_tiles = x[trial[:, None, None], times].permute(0, 1, 3, 2)
        target = self.bucketize(clean_tiles.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).detach()
        return dict(trial=trial, times=times, starts=starts, perm=perm, roll=roll,
                    raw_gain=gain(1), order_gain=gain(s["order_augment_scale"]), target=target)

    def ssl_loss(self, preprocessed, plan):
        if not bool(self.quantiles_ready):
            raise RuntimeError("Fit reconstruction quantiles on TRAIN data before training.")
        s = self.spec; w, k = s["window"], s["tiles"]
        tiles = preprocessed[plan["trial"][:, None, None], plan["times"]].permute(0, 1, 3, 2)
        n = len(tiles)
        zraw = self.encoder((tiles * plan["raw_gain"]).reshape(-1, self.channels, w))
        logits = self.reconstruct_head(zraw)
        recon = F.cross_entropy(logits.reshape(-1, s["levels"]), plan["target"].reshape(-1))
        zero = recon.new_zeros(())
        parts = dict(reconstruct=recon, position=zero, pair=zero, lag=zero)
        total = s["reconstruct"] * recon
        if s["order"] or s["pair"] or s["lag"]:
            view = tiles.gather(3, plan["roll"][:, :, None].expand(-1, -1, self.channels, -1))
            view = _tile_normalize(view * plan["order_gain"], "mean")
            labels = plan["perm"]
            view = view.gather(1, labels[..., None, None].expand_as(view))
            z = self.encoder(view.reshape(-1, self.channels, w)).reshape(n, k, -1)
            pos, score = self.order_head(z)
            pos_each = F.cross_entropy(pos.reshape(-1, k), labels.reshape(-1), reduction="none").reshape(n, k).mean(1)
            target, mask = _pair_targets(labels)
            pair_each = F.binary_cross_entropy_with_logits(
                (score[:, :, None] - score[:, None, :])[mask], target[mask], reduction="none").reshape(n, -1).mean(1)
            lag_each = torch.zeros(n, device=z.device)
            if s["lag"]:
                lag_target, lag_mask = lag_targets(plan["starts"], labels, self.lag_boundaries, s["lag_classes"])
                lag_logits = self.lag_head((z[:, :, None] - z[:, None])[lag_mask])
                lag_each = F.cross_entropy(lag_logits, lag_target, reduction="none").reshape(n, -1).mean(1)
            combined = s["order"] * pos_each + s["pair"] * pair_each + s["lag"] * lag_each
            keep = min(n, max(2, round(s["hard_fraction"] * n)))
            chosen = combined.detach().topk(keep).indices
            total = total + combined[chosen].mean()
            parts.update(position=pos_each.mean(), pair=pair_each.mean(), lag=lag_each.mean())
        parts["total"] = total
        return parts

    @torch.no_grad()
    def maybe_reset_heads(self, ssl_step, optimizer):
        every = self.spec["reset_every"]
        if not every or ssl_step % every:
            return
        generator = torch.Generator().manual_seed(self.args["seed"] + 7919 + ssl_step)
        for module in (self.order_head, self.lag_head):
            for name, p in module.named_parameters():
                if p.dim() >= 2:
                    bound = 1 / math.sqrt(p.shape[1])
                    value = torch.empty(p.shape).uniform_(-bound, bound, generator=generator)
                    p.copy_(value.to(p.device))
                elif name.endswith("bias"):
                    p.zero_()
                else:
                    p.fill_(1)
                # Unlike the old standalone runner, also clear stale Adam moments.
                optimizer.state.pop(p, None)
