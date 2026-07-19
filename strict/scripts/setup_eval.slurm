#!/bin/bash
# One-time Hopper setup for BabyLM Strict eval.
# Installs strict/requirements.txt into the shared ebdlm-babylm uv .venv and
# downloads evaluation_data onto $SCRATCH (visible to all later GPU eval jobs).
#
# Before submitting, put your Hugging Face token in a single-line file:
#   mkdir -p ${SCRATCH}/secrets
#   printf '%s\n' 'hf_...' > ${SCRATCH}/secrets/hf_token
#   chmod 600 ${SCRATCH}/secrets/hf_token
#
# Prerequisite: ${SCRATCH}/ebdlm-babylm/.venv from setup_hopper_env.slurm
#               ${SCRATCH}/babylm-eval cloned
#
#   sbatch strict/scripts/setup_eval.slurm
#
#SBATCH --partition=normal
#SBATCH --job-name=setup-babylm-eval
#SBATCH --output=setup-babylm-eval-%j.out
#SBATCH --error=setup-babylm-eval-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
EVAL_REPO="${EVAL_REPO:-${SCRATCH}/babylm-eval}"
EBDLM_REPO="${EBDLM_REPO:-${SCRATCH}/ebdlm-babylm}"
VENV="${VENV:-${EBDLM_REPO}/.venv}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${SCRATCH}/secrets/hf_token}"

export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/uv_cache}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${UV_CACHE_DIR}"

module load hosts/hopper gnu12/12.3.0 python/3.11.7-cx

PYTHON="${VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "Missing ${PYTHON}; run ebdlm-babylm scripts/setup_hopper_env.slurm first" >&2
  exit 1
fi

if [ ! -f "${HF_TOKEN_FILE}" ]; then
  echo "Missing HF token file: ${HF_TOKEN_FILE}" >&2
  echo "Create it (one line, chmod 600), then resubmit." >&2
  exit 1
fi
export HF_TOKEN
HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
if [ -z "${HF_TOKEN}" ]; then
  echo "HF token file is empty: ${HF_TOKEN_FILE}" >&2
  exit 1
fi

cd "${EVAL_REPO}/strict"

echo "Installing ${EVAL_REPO}/strict/requirements.txt into ${VENV}"
export PATH="${HOME}/.local/bin:${PATH}"
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt --python "${PYTHON}"
else
  "${PYTHON}" -m pip install -r requirements.txt
fi

echo "Downloading evaluation_data (scripts.download_evals)"
"${PYTHON}" -m scripts.download_evals

if [ -f evaluation_data/fast_eval/ewok_fast.zip ] && [ ! -d evaluation_data/fast_eval/ewok_fast ]; then
  echo "Unzipping ewok_fast.zip"
  unzip -o -P BabyLM2025 evaluation_data/fast_eval/ewok_fast.zip -d evaluation_data/fast_eval
fi

for p in \
  evaluation_data/fast_eval/blimp_fast \
  evaluation_data/fast_eval/supplement_fast \
  evaluation_data/fast_eval/entity_tracking_fast \
  evaluation_data/fast_eval/reading/reading_data.csv \
  evaluation_data/fast_eval/ewok_fast
do
  if [ ! -e "${p}" ]; then
    echo "Missing after download: ${p}" >&2
    exit 1
  fi
done

echo "OK: eval deps in ${VENV}; data under ${EVAL_REPO}/strict/evaluation_data"
