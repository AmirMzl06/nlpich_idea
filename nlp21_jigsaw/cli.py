import argparse
import json
import pickle
from pathlib import Path
import time
import torch
from .config import PRESETS
from .data import read_samples, make_loader, TrainingStream, fit_quantiles, CHARS
from .model import JigsawCTC
from .engine import (seed_all, move_batch, train_step, make_optimizer, set_learning_rates,
                     evaluate, atomic_save, save_checkpoint, load_checkpoint, restore_rng, json_write)


def train_parser():
    p = argparse.ArgumentParser(description="NLP21: Jigsaw pretrain/freeze or joint Jigsaw + CTC")
    p.add_argument("--jigsaw", action="store_true", help="Dispatch flag used by start_trainer.py")
    p.add_argument("--datasetPath", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--jigsaw_arm", choices=PRESETS, default="proposed")
    p.add_argument("--train_mode", choices=("frozen", "joint"), required=True)
    p.add_argument("--noise_site", choices=("neural", "embedding"), required=True)
    p.add_argument("--noise_std", "--whiteNoiseSD", dest="noise_std", type=float, default=0.8,
                   help="Additive Gaussian standard deviation; 0 disables it")
    p.add_argument("--source_augment", action="store_true",
                   help="Also enable the original Jigsaw neuron-dropout/gain/roll augmentations")
    p.add_argument("--gru", action="store_true")
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--gauss_in", action="store_true", help="Compatibility flag; smoothing is inside this model")
    p.add_argument("--no_gauss", action="store_true")
    p.add_argument("--batchSize", type=int, default=16)
    p.add_argument("--nBatch", type=int, default=20000, help="CTC/joint optimizer updates, not epochs")
    p.add_argument("--pretrain_steps", type=int, default=6000, help="SSL updates only for frozen mode, not epochs")
    p.add_argument("--ssl_batch", type=int, default=64, help="Jigsaw spans per optimizer update")
    p.add_argument("--ssl_weight", type=float, default=1.0)
    p.add_argument("--ssl_lr", type=float, default=1e-3)
    p.add_argument("--jigsaw_dim", type=int, default=64)
    p.add_argument("--jigsaw_width", type=int, default=64)
    p.add_argument("--head_width", type=int, default=64)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--kernel", type=int, default=32)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--nInputFeatures", type=int, default=192)
    p.add_argument("--lrStart", type=float, default=0.02)
    p.add_argument("--lrEnd", type=float, default=0.002)
    p.add_argument("--l2_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantile_samples", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    # Accepted only for copy/paste compatibility with the old CEBRA command.
    p.add_argument("--offset", type=int, default=None)
    p.add_argument("--random_offset", action="store_true")
    return p


def validate_args(p, a):
    positive = ("batchSize", "nBatch", "pretrain_steps", "ssl_batch", "jigsaw_dim", "jigsaw_width",
                "head_width", "hidden", "layers", "kernel", "stride", "nInputFeatures",
                "quantile_samples", "eval_every", "save_every", "log_every")
    for key in positive:
        if a[key] < 1:
            p.error(f"--{key} must be positive")
    for key in ("ssl_lr", "ssl_weight", "lrStart", "lrEnd", "grad_clip"):
        if a[key] <= 0:
            p.error(f"--{key} must be positive")
    for key in ("noise_std", "l2_decay", "num_workers", "seed"):
        if a[key] < 0:
            p.error(f"--{key} cannot be negative")
    if not 0 <= a["dropout"] < 1:
        p.error("--dropout must be in [0, 1)")


def resolve_device(name):
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def train_main(argv=None):
    p = train_parser()
    args = vars(p.parse_args(argv))
    validate_args(p, args)
    args["model_type"] = "nlp21_jigsaw_v1"
    out = Path(args["out_dir"])
    config_path = out / "config.json"
    checkpoint_path = out / "checkpoint.pt"
    if out.exists() and any(out.iterdir()) and not args["resume"]:
        p.error(f"{out} is not empty. Choose a new out_dir or use --resume.")
    if args["resume"] and not checkpoint_path.is_file():
        p.error(f"Cannot resume: {checkpoint_path} is missing")
    out.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint_path) if args["resume"] else None
    if state is not None:
        mutable = {"resume", "datasetPath", "out_dir", "device", "num_workers", "log_every", "eval_every", "save_every", "jigsaw"}
        mismatches = {k: (state["config"].get(k), v) for k, v in args.items()
                      if k not in mutable and state["config"].get(k) != v}
        if mismatches:
            p.error(f"Resume config mismatch: {mismatches}")
    seed_all(args["seed"])
    device = resolve_device(args["device"])
    model = JigsawCTC(args).to(device)
    print(json.dumps(dict(device=device, arm=args["jigsaw_arm"], train_mode=args["train_mode"],
                          noise_site=args["noise_site"], noise_std=args["noise_std"],
                          ctc_updates=args["nBatch"], pretrain_updates=args["pretrain_steps"] if args["train_mode"] == "frozen" else 0,
                          preset=model.spec), indent=2), flush=True)
    if args["offset"] is not None or args["random_offset"]:
        print("NOTE: --offset and --random_offset are CEBRA-only and unused here. "
              "Jigsaw gaps come from the selected preset.", flush=True)
    train_samples = read_samples(args["datasetPath"], "train")
    validation_samples = read_samples(args["datasetPath"], "heldout")
    for split, samples in (("train", train_samples), ("heldout", validation_samples)):
        if any(x.shape[1] != model.channels for x, _, _ in samples):
            raise ValueError(f"Feature count mismatch in {split}; expected {model.channels}")
        for x, text, _ in samples:
            if not torch.isfinite(x).all():
                raise ValueError(f"Nonfinite neural data in {split}")
    stream = TrainingStream(train_samples, args["batchSize"], args["seed"] + 71)
    valid_loader = make_loader(validation_samples, args["batchSize"], args["num_workers"])
    if state is None:
        fit_quantiles(model, train_samples, args["quantile_samples"], args["seed"])
    else:
        model.load_state_dict(state["model"])
        stream.load_state_dict(state["stream"])
    json_write(config_path, args)
    json_write(out / "experiment.json", dict(config=args, preset=model.spec, charset=CHARS,
                train_trials=len(train_samples), validation_trials=len(validation_samples),
                validation_split="heldout", final_eval_split="all", final_checkpoint="last",
                normalization="original get_input: per-file, per-block, includes unlabeled held-out block statistics"))
    with (out / "args").open("wb") as f:
        pickle.dump(args, f)
    stages = [("pretrain", args["pretrain_steps"]), ("decoder", args["nBatch"])] if args["train_mode"] == "frozen" else [("joint", args["nBatch"])]
    best_cer = state["best_cer"] if state is not None else float("inf")
    waiting = state is not None
    for stage, budget in stages:
        if stage == "decoder":
            model.freeze_encoder()
        if waiting and stage != state["stage"]:
            continue
        optimizer = make_optimizer(model, args, stage)
        start = 0
        if waiting:
            optimizer.load_state_dict(state["optimizer"])
            # torch.load was CPU-based; Adam state follows parameter devices.
            for param, values in optimizer.state.items():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        values[key] = value.to(param.device)
            start = state["step"]
            restore_rng(state)
            waiting = False
        print(f"STAGE {stage}: updates {start} -> {budget}; encoder_frozen={model.encoder_frozen}", flush=True)
        began = time.monotonic()
        for step in range(start, budget):
            set_learning_rates(optimizer, step, budget, args)
            if stage in ("pretrain", "joint"):
                model.maybe_reset_heads(step + 1, optimizer)
            batch = move_batch(stream.next(), device)
            metrics = train_step(model, optimizer, batch, args, stage)
            row = dict(stage=stage, step=step + 1, **metrics)
            if (step + 1) % args["log_every"] == 0 or step == start or step + 1 == budget:
                row["elapsed_seconds"] = time.monotonic() - began
                print(json.dumps(row), flush=True)
                with (out / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if stage != "pretrain" and ((step + 1) % args["eval_every"] == 0 or step + 1 == budget):
                validation, _, _ = evaluate(model, valid_loader, device)
                report = dict(stage=stage, step=step + 1, split="heldout", **validation)
                print("VALID " + json.dumps(report), flush=True)
                with (out / "validation.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                if validation["cer"] < best_cer:
                    best_cer = validation["cer"]
                    atomic_save(model.state_dict(), out / "modelWeights.best")
            if (step + 1) % args["save_every"] == 0 or step + 1 == budget:
                save_checkpoint(checkpoint_path, model, optimizer, args, stage, step + 1, stream, best_cer)
                if stage != "pretrain":
                    atomic_save(model.state_dict(), out / "modelWeights")
        if stage == "pretrain":
            atomic_save(dict(config=args, model=model.state_dict()), out / "pretrained.pt")
        else:
            # Also refresh after resuming a completed checkpoint, in case the
            # previous process stopped between checkpoint and weight-file writes.
            atomic_save(model.state_dict(), out / "modelWeights")
    json_write(out / "training_complete.json", dict(status="complete", config=args,
                                                    completed_stages=stages, best_heldout_cer=best_cer))
    print(f"Training complete: {out}. Run eval_single_model.py --out_dir {out} --datasetPath ...", flush=True)


def eval_main(argv=None):
    p = argparse.ArgumentParser(description="Clean, greedy CTC CER for a saved NLP21 Jigsaw model")
    p.add_argument("--datasetPath", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--split", choices=("all", "heldout", "online"), default="all")
    p.add_argument("--checkpoint", choices=("last", "best"), default="last")
    p.add_argument("--save_logits", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    args = json.loads((out / "config.json").read_text())
    device = resolve_device(a.device)
    model = JigsawCTC(args).to(device)
    weight_path = out / ("modelWeights" if a.checkpoint == "last" else "modelWeights.best")
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    loader = make_loader(read_samples(a.datasetPath, a.split), a.batch_size)
    result, rows, outputs = evaluate(model, loader, device, a.save_logits)
    result.update(arm=args["jigsaw_arm"], train_mode=args["train_mode"], noise_site=args["noise_site"],
                  seed=args["seed"], split=a.split, checkpoint=a.checkpoint,
                  noise_std=args["noise_std"], evaluation="clean_greedy_ctc", out_dir=str(out))
    report_path = out / f"eval_{a.split}_{a.checkpoint}.json"
    json_write(report_path, result)
    json_write(out / f"predictions_{a.split}_{a.checkpoint}.json", rows)
    with (out / f"evalStats_{a.split}_{a.checkpoint}.pkl").open("wb") as f:
        pickle.dump([(r["errors"], r["characters"]) for r in rows], f)
    if outputs:
        # Original class ordering; explicitly no implicit external LM remapping.
        atomic_save(dict(charset=["<BLANK>"] + CHARS, outputs=outputs), out / f"logits_{a.split}_{a.checkpoint}.pt")
    print(json.dumps(result, indent=2), flush=True)
    print(f"CER: {result['cer']:.6f} ({result['cer_percent']:.3f}%); saved to {report_path}", flush=True)
