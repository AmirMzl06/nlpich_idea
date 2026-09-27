#!/usr/bin/env bash
set -euo pipefail

# Run on your RunAI login machine. The code folder must be in a mounted PVC.
project_dir=${1:?Usage: bash submit_nlp21_jigsaw.sh /absolute/code/path [job_name] [noise_std]}
job_name=${2:-mm8-nlp21-jigsaw}
noise_std=${3:-0.8}
if [[ "$project_dir" != /* ]]; then
    echo "Pass the absolute code path as seen inside the job." >&2
    exit 2
fi

python_args=(python -u run_jigsaw_ctc_matrix.py)
python_args+=(--datasetPath /data/hossein/mm_project/CORP_data_release)
python_args+=(--out_root "nlp21-jigsaw-gauss${noise_std}")
python_args+=(--noise_std "$noise_std" --pretrain_steps 6000 --nBatch 20000 --seed 0)
printf -v python_command '%q ' "${python_args[@]}"
printf -v quoted_project '%q' "$project_dir"
job_command="source /home/mirzaei/lm-cebra/.venv/bin/activate && cd $quoted_project && exec $python_command"

runai_args=(submit --name "$job_name")
runai_args+=(--image registry.rcp.epfl.ch/upmwmathis-mirzaei/robust-cebra:v0.4)
runai_args+=(--gpu 1 --cpu 64 --memory 256Gi --node-pools h200 --large-shm)
runai_args+=(--pvc home:/home/mirzaei --pvc upmwmathis-scratch:/data)
runai_args+=(-e USER=mirzaei -e LOGNAME=mirzaei)
runai_args+=(-e LD_LIBRARY_PATH=/home/mirzaei/lm-cebra/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64)
runai_args+=(-p upmwmathis-mirzaei --run-as-user --command -- bash -lc "$job_command")
runai "${runai_args[@]}"
