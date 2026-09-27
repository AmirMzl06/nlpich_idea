"""Run 4 Gaussian-noise training cells per arm, then collect CLEAN CER."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from nlp21_jigsaw.config import PRESETS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasetPath", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--arms", nargs="+", choices=PRESETS, default=["proposed", "order_lag", "order_time_axis", "order_full"])
    p.add_argument("--train_modes", nargs="+", choices=("frozen", "joint"), default=["frozen", "joint"])
    p.add_argument("--noise_sites", nargs="+", choices=("neural", "embedding"), default=["neural", "embedding"])
    p.add_argument("--noise_std", type=float, default=0.8)
    p.add_argument("--nBatch", type=int, default=20000)
    p.add_argument("--pretrain_steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", choices=("both", "train", "eval"), default="both")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a, extra = p.parse_known_args()
    reserved = {"--jigsaw_arm", "--train_mode", "--noise_site", "--out_dir", "--jigsaw"}
    if any(v.split("=")[0] in reserved for v in extra):
        p.error("Matrix cell arguments are controlled by --arms/--train_modes/--noise_sites")
    if a.noise_std < 0:
        p.error("--noise_std must be nonnegative")
    root = Path(a.out_root).resolve()
    code = Path(__file__).resolve().parent
    records = []
    total = len(a.arms) * len(a.train_modes) * len(a.noise_sites)
    print(f"{total} runs; Gaussian std={a.noise_std}; clean final CER", flush=True)
    for arm in a.arms:
        for mode in a.train_modes:
            for site in a.noise_sites:
                out = root / arm / f"{mode}_{site}_s{a.seed}"
                train = [sys.executable, "-u", str(code / "start_trainer.py"), "--jigsaw",
                         "--datasetPath", a.datasetPath, "--out_dir", str(out), "--jigsaw_arm", arm,
                         "--train_mode", mode, "--noise_site", site, "--noise_std", str(a.noise_std),
                         "--pretrain_steps", str(a.pretrain_steps), "--nBatch", str(a.nBatch), "--seed", str(a.seed),
                         "--gru", "--gauss_in", "--bidir", "--batchSize", "16", "--hidden", "1024",
                         "--dropout", "0.4", "--layers", "5", "--kernel", "32", "--stride", "4", *extra]
                if a.resume and (out / "checkpoint.pt").exists():
                    train.append("--resume")
                evaluation = [sys.executable, "-u", str(code / "eval_single_model.py"),
                              "--datasetPath", a.datasetPath, "--out_dir", str(out)]
                # Forward the requested device to evaluation as well.
                for i, v in enumerate(extra):
                    if v == "--device" and i + 1 < len(extra):
                        evaluation += ["--device", extra[i + 1]]
                    elif v.startswith("--device="):
                        evaluation.append(v)
                commands = ([train] if a.run == "train" else [evaluation] if a.run == "eval" else [train, evaluation])
                for command in commands:
                    print(shlex.join(command), flush=True)
                    if not a.dry_run:
                        subprocess.run(command, cwd=code, check=True)
                if not a.dry_run and a.run != "train":
                    records.append(json.loads((out / "eval_all_last.json").read_text()))
                    write_summary(root, records)
    if records:
        print((root / "cer_table.md").read_text(), flush=True)


def write_summary(root, records):
    root.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "train_mode", "noise_site", "noise_std", "seed", "cer", "cer_percent", "ctc_loss",
              "errors", "characters", "trials", "split", "checkpoint", "out_dir"]
    with (root / "cer_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    columns = [(m, s) for m in ("frozen", "joint") for s in ("neural", "embedding")]
    lines = ["| Arm | Frozen + neural | Frozen + embedding | Joint + neural | Joint + embedding |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in dict.fromkeys(r["arm"] for r in records):
        cells = {(r["train_mode"], r["noise_site"]): f"{r['cer_percent']:.3f}%" for r in records if r["arm"] == arm}
        lines.append("| " + " | ".join([arm] + [cells.get(key, "—") for key in columns]) + " |")
    (root / "cer_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
