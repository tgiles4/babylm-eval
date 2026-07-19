#!/bin/bash
# BabyLM Strict fast zero-shot eval on Hopper (1× A100 80GB).
# Thin job: assumes setup_eval.slurm already ran once.
#
# Prerequisites (on $SCRATCH):
#   - babylm-eval + evaluation_data/   (setup_eval.slurm)
#   - ebdlm-babylm/.venv with requirements.txt installed (setup_eval.slurm)
#   - checkpoints: ebdlm-babylm/runs/<run>/hf/<revision>
#
#   sbatch strict/scripts/eval_zero_shot_fast.slurm <run_name> <revision> <backend>
#   sbatch --job-name=eval-gc28-diff strict/scripts/eval_zero_shot_fast.slurm good-cloud-28 last diffusion
#
# Results: strict/results/<run_name>/<revision>/zero_shot/<backend>/...
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
#SBATCH --time=00:45:00

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
EVAL_REPO="${EVAL_REPO:-${SCRATCH}/babylm-eval}"
EBDLM_REPO="${EBDLM_REPO:-${SCRATCH}/ebdlm-babylm}"
VENV="${VENV:-${EBDLM_REPO}/.venv}"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch strict/scripts/eval_zero_shot_fast.slurm <run_name> <revision> <backend> [eval_dir]" >&2
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
export MC_NUM="${MC_NUM:-128}"
export MC_BATCH_SIZE="${MC_BATCH_SIZE:-128}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

echo "RUN_NAME=${RUN_NAME} REVISION=${REVISION} BACKEND=${BACKEND}"
echo "PYTHON=${PYTHON} BATCH_SIZE=${BATCH_SIZE} MC_NUM=${MC_NUM} MC_BATCH_SIZE=${MC_BATCH_SIZE}"

srun --export=ALL env \
  PATH="$(dirname "${PYTHON}"):${PATH}" \
  bash scripts/eval_zero_shot_fast.sh \
    "${MODEL_PATH}" \
    "${REVISION}" \
    "${BACKEND}" \
    "${EVAL_DIR_REL}"
