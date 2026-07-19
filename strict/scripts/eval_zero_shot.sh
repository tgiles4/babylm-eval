#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=${3:-"evaluation_data/full_eval"}
REVISION=${4:-main}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo $BACKEND_READ

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

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions --revision_name $REVISION --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions --revision_name $REVISION --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions --revision_name $REVISION --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions --revision_name $REVISION --batch_size $BATCH_SIZE $EXTRA_ARGS
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task comps --data_path "${EVAL_DIR}/comps" --save_predictions --revision_name $REVISION --batch_size $BATCH_SIZE $EXTRA_ARGS

if [[ "$BACKEND" != "energy" ]]; then
    python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND_READ --data_path "${EVAL_DIR}/reading/reading_data.csv" --revision_name $REVISION $EXTRA_ARGS
fi
