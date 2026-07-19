#!/bin/bash

MODEL_PATH=$1
REVISION_NAME=$2
BACKEND=$3
EVAL_DIR=${4:-"evaluation_data/fast_eval"}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

MC_NUM=${MC_NUM:-128}
MC_BATCH_SIZE=${MC_BATCH_SIZE:-16}
EBDLM_ROOT=${EBDLM_ROOT:-"${HOME}/ebdlm-babylm"}
if [[ "$BACKEND" == "diffusion" || "$BACKEND" == "energy" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-8}
else
    BATCH_SIZE=${BATCH_SIZE:-64}
fi

EXTRA_ARGS=""
if [[ "$BACKEND" == "diffusion" || "$BACKEND" == "energy" ]]; then
    EXTRA_ARGS="--mc_num ${MC_NUM} --mc_batch_size ${MC_BATCH_SIZE}"
fi
if [[ "$BACKEND" == "energy" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --ebdlm_root ${EBDLM_ROOT}"
fi

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_fast" --save_predictions --revision_name $REVISION_NAME --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_fast" --save_predictions --revision_name $REVISION_NAME --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_fast" --save_predictions --revision_name $REVISION_NAME --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking_fast" --save_predictions --revision_name $REVISION_NAME --batch_size $BATCH_SIZE $EXTRA_ARGS

# Reading: diffusion uses Eq.6 surprisal; energy reading not wired yet.
if [[ "$BACKEND" != "energy" ]]; then
    python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND_READ --data_path "${EVAL_DIR}/reading/reading_data.csv" --revision_name $REVISION_NAME $EXTRA_ARGS
fi
