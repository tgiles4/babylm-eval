#!/bin/bash
# BabyLM Strict fast zero-shot eval on Hopper (1× A100 80GB).
# Supports backends: diffusion | energy | mlm | causal | mntp | enc_dec_*
#
# Prerequisites (on $SCRATCH):
#   - this repo cloned: ${SCRATCH}/babylm-eval
#   - HF checkpoints:   ${SCRATCH}/ebdlm-babylm/runs/<run>/hf/<revision>
#   - Python env with torch + transformers (reuse ebdlm .venv or make strict/.venv)
#   - evaluation_data under strict/ (download once on a login/CPU job if needed)
#
# Submit from babylm-eval repo root (or anywhere; script cds to SCRATCH copy):
#   sbatch strict/scripts/eval_zero_shot_fast.slurm <run_name> <revision> <backend>
#   sbatch --job-name=eval-gc28-diff strict/scripts/eval_zero_shot_fast.slurm good-cloud-28 last diffusion
#   sbatch --job-name=eval-ag25-diff strict/scripts/eval_zero_shot_fast.slurm avid-gorge-25 last diffusion
#
# Results: strict/results/<run_name>/<revision>/zero_shot/<backend>/...
# A100 defaults: BATCH_SIZE=64, MC_BATCH_SIZE=128 (override via env if needed).
#
#SBATCH --job-name=babylm-eval-fast
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --constraint=amd
#SBATCH --output=babylm-eval-fast-%j.out
#SBATCH --error=babylm-eval-fast-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
EVAL_REPO="${EVAL_REPO:-${SCRATCH}/babylm-eval}"
EBDLM_REPO="${EBDLM_REPO:-${SCRATCH}/ebdlm-babylm}"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch strict/scripts/eval_zero_shot_fast.slurm <run_name> <revision> <backend> [eval_dir]" >&2
  echo "Example: sbatch strict/scripts/eval_zero_shot_fast.slurm good-cloud-28 last diffusion" >&2
  exit 1
fi

RUN_NAME="$1"
REVISION="$2"
BACKEND="$3"
EVAL_DIR_REL="${4:-evaluation_data/fast_eval}"

MODEL_PATH="${EBDLM_REPO}/runs/${RUN_NAME}/hf"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing HF export dir: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -d "${MODEL_PATH}/${REVISION}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "Missing revision '${REVISION}' under ${MODEL_PATH}" >&2
  exit 1
fi

MODEL_PATH_OR_NAME="${MODEL_PATH}"
REVISION_NAME="${REVISION}"

cd "${EVAL_REPO}/strict"

module load hosts/hopper gnu12/12.3.0 cuda/12.6.2-ou python/3.11.7-cx

# Prefer eval-local venv; fall back to ebdlm training venv (has torch + flash-attn).
if [ -x "${EVAL_REPO}/strict/.venv/bin/python" ]; then
  PYTHON="${EVAL_REPO}/strict/.venv/bin/python"
elif [ -x "${EBDLM_REPO}/.venv/bin/python" ]; then
  PYTHON="${EBDLM_REPO}/.venv/bin/python"
else
  echo "No venv found at ${EVAL_REPO}/strict/.venv or ${EBDLM_REPO}/.venv" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${EVAL_REPO}/strict${PYTHONPATH:+:${PYTHONPATH}}"
export EBDLM_ROOT="${EBDLM_REPO}"

# A100 80GB: pack the full MC budget in one forward; larger example batches.
export MC_NUM="${MC_NUM:-128}"
export MC_BATCH_SIZE="${MC_BATCH_SIZE:-128}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}"
echo "REVISION_NAME=${REVISION_NAME}"
echo "BACKEND=${BACKEND}"
echo "BATCH_SIZE=${BATCH_SIZE} MC_NUM=${MC_NUM} MC_BATCH_SIZE=${MC_BATCH_SIZE}"
echo "PYTHON=${PYTHON}"
echo "EBDLM_ROOT=${EBDLM_ROOT}"

srun --export=ALL env \
  PATH="$(dirname "${PYTHON}"):${PATH}" \
  bash scripts/eval_zero_shot_fast.sh \
    "${MODEL_PATH_OR_NAME}" \
    "${REVISION_NAME}" \
    "${BACKEND}" \
    "${EVAL_DIR_REL}"
