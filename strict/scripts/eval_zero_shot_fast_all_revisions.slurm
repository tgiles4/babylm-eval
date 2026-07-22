#!/bin/bash
# BabyLM Strict fast zero-shot eval across all chck_*M revisions on Hopper (1× A100 80GB).
# Thin job: assumes setup_eval.slurm already ran once.
#
# Prerequisites (on $SCRATCH):
#   - babylm-eval + evaluation_data/   (setup_eval.slurm)
#   - ebdlm-babylm/.venv with requirements.txt installed (setup_eval.slurm)
#   - checkpoints: ebdlm-babylm/runs/<run>/hf/chck_*M
#
#   sbatch strict/scripts/eval_zero_shot_fast_all_revisions.slurm <run_name> <backend> <track>
#   sbatch --job-name=eval-bt1152-all \
#     strict/scripts/eval_zero_shot_fast_all_revisions.slurm backbone-thin-1152 diffusion strict
#
# Track: strict (chck_1M..chck_1000M) | strict-small (stops at chck_100M)
# Does not evaluate the `last` revision — use eval_zero_shot_fast.slurm for that.
# Results: strict/results/<run_name>/<revision>/zero_shot/<backend>/...
#
# Walltime: ~28 checkpoints × ~18 min ≈ 8.5 h for full strict (from job 9042869);
# default 12 h leaves headroom. Override with --time if needed.
#
#SBATCH --job-name=babylm-eval-fast-all
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --constraint=amd
#SBATCH --output=babylm-eval-fast-all-%j.out
#SBATCH --error=babylm-eval-fast-all-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
EVAL_REPO="${EVAL_REPO:-${SCRATCH}/babylm-eval}"
EBDLM_REPO="${EBDLM_REPO:-${SCRATCH}/ebdlm-babylm}"
VENV="${VENV:-${EBDLM_REPO}/.venv}"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch strict/scripts/eval_zero_shot_fast_all_revisions.slurm <run_name> <backend> <track> [eval_dir]" >&2
  echo "  track: strict | strict-small" >&2
  exit 1
fi

RUN_NAME="$1"
BACKEND="$2"
TRACK="$3"
EVAL_DIR_REL="${4:-evaluation_data/fast_eval}"

if [[ "${TRACK}" != "strict" && "${TRACK}" != "strict-small" ]]; then
  echo "track must be 'strict' or 'strict-small', got: ${TRACK}" >&2
  exit 1
fi

MODEL_PATH="${EBDLM_REPO}/runs/${RUN_NAME}/hf"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing HF export dir: ${MODEL_PATH}" >&2
  exit 1
fi

PYTHON="${VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "Missing ${PYTHON}; run setup_eval.slurm / setup_hopper_env.slurm first" >&2
  exit 1
fi

cd "${EVAL_REPO}/strict"

if [ ! -d evaluation_data/fast_eval/blimp_fast ]; then
  echo "Missing evaluation_data; run: sbatch strict/scripts/setup_eval.slurm" >&2
  exit 1
fi

module load hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${EVAL_REPO}/strict${PYTHONPATH:+:${PYTHONPATH}}"
export EBDLM_ROOT="${EBDLM_REPO}"
# Diffusion GPU batch ≈ BATCH_SIZE × MC_BATCH_SIZE.
# batch_size=64 × mc_batch_size=128 OOMed on A100.80GB (~41 GiB extra alloc);
# 32 × 128 = 4096 rows fits with headroom for padded BLiMP/EWoK batches.
export MC_NUM="${MC_NUM:-128}"
export MC_BATCH_SIZE="${MC_BATCH_SIZE:-128}"
export BATCH_SIZE="${BATCH_SIZE:-32}"

echo "RUN_NAME=${RUN_NAME} BACKEND=${BACKEND} TRACK=${TRACK}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PYTHON=${PYTHON} BATCH_SIZE=${BATCH_SIZE} MC_NUM=${MC_NUM} MC_BATCH_SIZE=${MC_BATCH_SIZE} (gpu_batch≈$((BATCH_SIZE * MC_BATCH_SIZE)))"

srun --export=ALL env \
  PATH="$(dirname "${PYTHON}"):${PATH}" \
  bash scripts/eval_zero_shot_fast_all_revisions.sh \
    "${MODEL_PATH}" \
    "${BACKEND}" \
    "${TRACK}" \
    "${EVAL_DIR_REL}"
