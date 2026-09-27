"""Four explicit training cells and clean CER evaluation."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from .noise import gaussian_noise
from .model import time_mask


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ctc_loss(logits, lengths, y, yl):
    # Reject impossible alignments instead of silently zeroing their gradients.
    valid_pairs = torch.arange(max(0, y.shape[1] - 1), device=y.device)[None] < (yl.to(y.device) - 1)[:, None]
    repeats = ((y[:, 1:] == y[:, :-1]) & valid_pairs).sum(1)
    required = yl.to(y.device) + repeats
    if (lengths.to(y.device) < required).any():
        raise ValueError("CTC alignment impossible: output length < target length + adjacent repeats. "
                         "Reduce --stride/--kernel or inspect the transcripts.")
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), y,
                      lengths.cpu().long(), yl.cpu().long(), blank=0,
                      reduction="mean", zero_infinity=False)


def move_batch(batch, device):
    return tuple(t.to(device) for t in batch)


def train_step(model, optimizer, batch, args, stage):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    clean, y, lengths, yl, _ = batch
    do_ssl = stage in ("pretrain", "joint")
    do_ctc = stage in ("decoder", "joint")
    # Neural noise is applied while the encoder is being trained. In joint mode
    # the SAME noisy neural input feeds both SSL and the CTC path.
    x = gaussian_noise(clean, lengths, args["noise_std"]) if (
        args["noise_site"] == "neural" and do_ssl) else clean
    parts, losses = {}, []
    if do_ssl:
        with torch.no_grad():
            clean_view = model.preprocess(clean, lengths)
            plan = model.sample_plan(clean_view, lengths, args["ssl_batch"])
        ssl = model.ssl_loss(model.preprocess(x, lengths), plan)
        parts.update({f"ssl_{k}": float(v.detach()) for k, v in ssl.items()})
        losses.append(args["ssl_weight"] * ssl["total"])
    if do_ctc:
        if model.encoder_frozen:
            with torch.no_grad():
                z = model.encode(x, lengths)
        else:
            z = model.encode(x, lengths)
        if args["noise_site"] == "embedding":
            # Addition retains the gradient to the encoder in joint mode.
            z = gaussian_noise(z, lengths, args["noise_std"])
        logits, out_lengths = model.decode(z, lengths)
        ctc = ctc_loss(logits, out_lengths, y, yl)
        parts["ctc"] = float(ctc.detach())
        losses.append(ctc)
    total = sum(losses)
    if not torch.isfinite(total):
        raise FloatingPointError(f"Non-finite {stage} loss: {parts}")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], args["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    parts.update(total=float(total.detach()), grad_norm=float(norm.detach()))
    return parts


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        next_row = [i]
        for j, right in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = next_row
    return row[-1]


@torch.no_grad()
def evaluate(model, loader, device, include_logits=False):
    mode = model.training
    model.eval()
    errors = chars = trials = 0
    loss_sum = 0.0
    rows, outputs = [], []
    for batch in loader:
        x, y, lengths, yl, days = move_batch(batch, device)
        logits, out_lengths = model(x, lengths)
        loss = ctc_loss(logits, out_lengths, y, yl)
        loss_sum += float(loss) * len(x)
        trials += len(x)
        for i, length in enumerate(out_lengths.tolist()):
            ids = logits[i, :length].argmax(-1).unique_consecutive().tolist()
            ids = [v for v in ids if v != 0]
            target = y[i, :int(yl[i])].tolist()
            distance = edit_distance(target, ids)
            errors += distance; chars += len(target)
            rows.append(dict(errors=distance, characters=len(target), session=int(days[i]),
                             target=target, prediction=ids))
            if include_logits:
                outputs.append(dict(logits=logits[i, :length].float().cpu(), length=length, target=target))
    model.train(mode)
    if not chars:
        raise ValueError("Evaluation has no target characters")
    return dict(cer=errors / chars, cer_percent=100 * errors / chars,
                ctc_loss=loss_sum / trials, errors=errors, characters=chars, trials=trials), rows, outputs


def make_optimizer(model, args, stage):
    groups = []
    if stage in ("pretrain", "joint"):
        groups.append(dict(params=list(model.ssl_parameters()), lr=args["ssl_lr"],
                           initial_lr=args["ssl_lr"], eps=1e-8, weight_decay=0.0, name="ssl"))
    if stage in ("decoder", "joint"):
        groups.append(dict(params=list(model.decoder_parameters()), lr=args["lrStart"],
                           initial_lr=args["lrStart"], eps=0.1, weight_decay=args["l2_decay"], name="ctc"))
    return torch.optim.Adam(groups)


def set_learning_rates(optimizer, step, total, args):
    progress = step / max(1, total - 1)
    for group in optimizer.param_groups:
        if group["name"] == "ctc":
            group["lr"] = args["lrStart"] + progress * (args["lrEnd"] - args["lrStart"])
        else:
            group["lr"] = args["ssl_lr"]


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def save_checkpoint(path, model, optimizer, args, stage, step, stream, best_cer):
    atomic_save(dict(format="nlp21_jigsaw_v1", config=args, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), stage=stage, step=step,
                     stream=stream.state_dict(), best_cer=best_cer,
                     rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), path)


def load_checkpoint(path, device="cpu"):
    # Only local checkpoints produced by this program, not untrusted downloads.
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng(state):
    torch.set_rng_state(state["rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
