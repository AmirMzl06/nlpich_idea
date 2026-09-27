"""NLP21 splits and labels matching the supplied repository, without CEBRA imports."""
from pathlib import Path
import warnings
import torch
from torch.utils.data import DataLoader
from utils.data_loader import get_input

CHARS = ['>', ',', '?', '~', "'"] + list('abcdefghijklmnopqrstuvwxyz')
CHAR_TO_ID = {s: i + 1 for i, s in enumerate(CHARS)}


def read_samples(dataset_path, split):
    root = Path(dataset_path)
    seed = root / "seed_model_training_data/mat"
    def load(path, **kwargs):
        if not path.is_dir() or not any(path.rglob("*.mat")):
            raise FileNotFoundError(f"No NLP21 .mat files in {path}")
        # Same per-block normalization as the supplied get_input. Smoothing is
        # always inside the new model, including when gauss_in is not specified.
        return get_input(str(path), norm=True, gauss=False, **kwargs)
    if split == "train":
        samples = load(seed, train=True)
    elif split == "heldout":
        samples = load(seed, valid=True)
    elif split in ("online", "all"):
        online = []
        for branch in ("no_recalibration", "recalibration"):
            online.extend(load(root / f"online_evaluation_data/{branch}/mat"))
        samples = (load(seed, valid=True) if split == "all" else []) + online
    else:
        raise ValueError(f"Unknown split {split}")
    if not samples:
        raise ValueError(f"Empty {split} split")
    unknown = sorted({c for _, text, _ in samples for c in str(text) if c not in CHAR_TO_ID})
    if unknown:
        warnings.warn(f"Filtering characters absent from the original charset: {unknown!r}")
    return samples


def collate(samples):
    xs, texts, days = zip(*samples)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    x = torch.zeros(len(xs), int(lengths.max()), xs[0].shape[1])
    targets = [[CHAR_TO_ID[c] for c in str(text) if c in CHAR_TO_ID] for text in texts]
    yl = torch.tensor([len(t) for t in targets], dtype=torch.long)
    if (yl == 0).any():
        raise ValueError("Empty transcript after applying the NLP21 charset")
    y = torch.zeros(len(xs), int(yl.max()), dtype=torch.long)
    for i, (item, target) in enumerate(zip(xs, targets)):
        x[i, :len(item)] = item
        y[i, :len(target)] = torch.tensor(target)
    return x, y, lengths, yl, torch.tensor(days, dtype=torch.long)


def make_loader(samples, batch_size, workers=0):
    return DataLoader(samples, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate)


class TrainingStream:
    """Shuffled epochs with serializable cursor, independent of Gaussian RNG."""
    def __init__(self, samples, batch_size, seed):
        self.samples, self.batch_size = samples, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(samples), generator=self.generator)
        self.cursor = 0

    def next(self):
        if self.cursor == len(self.samples):
            self.order = torch.randperm(len(self.samples), generator=self.generator)
            self.cursor = 0
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(indices)
        return collate([self.samples[int(i)] for i in indices])

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, generator=self.generator.get_state(),
                    samples=len(self.samples))

    def load_state_dict(self, state):
        if state["samples"] != len(self.samples):
            raise ValueError("Training dataset size changed since the checkpoint")
        self.order, self.cursor = state["order"].cpu(), state["cursor"]
        self.generator.set_state(state["generator"].cpu())


@torch.no_grad()
def fit_quantiles(model, samples, limit, seed):
    # Uniform random TRAIN time bins, sampled without concatenating trials or
    # including padding. Reconstruction targets use the same smoothed domain.
    lengths = torch.tensor([len(x) for x, _, _ in samples])
    ends = lengths.cumsum(0)
    rng = torch.Generator().manual_seed(seed + 509)
    locations = torch.randint(int(ends[-1]), (min(limit, int(ends[-1])),), generator=rng)
    trials = torch.searchsorted(ends, locations, right=True)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), ends[:-1]))
    values = []
    device = next(model.parameters()).device
    for i in trials.unique().tolist():
        x = samples[i][0].to(device)
        smooth = model.preprocess(x[None], torch.tensor([len(x)], device=device))[0]
        values.append(smooth[(locations[trials == i] - offsets[i]).to(device)].cpu())
    model.set_quantiles(torch.cat(values).to(device))
