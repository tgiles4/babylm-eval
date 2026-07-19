#!/bin/bash
MODEL_NAME=$1
BACKEND=$2
TRACK_NAME=$3
WORD_PATH=${4:-"evaluation_data/full_eval/aoa/cdi_childes.json"}
OUTPUT_DIR=${5:-"results"}

# please check evaluation_pipeline/AoA_word/README.md for more information on parameter setting

# Set default parameters
MIN_CONTEXT=${MIN_CONTEXT:-20}
MC_NUM=${MC_NUM:-128}
MC_BATCH_SIZE=${MC_BATCH_SIZE:-16}
echo "Running AoA evaluation for model: $MODEL_NAME"
echo "Word path: $WORD_PATH"
echo "Output directory: $OUTPUT_DIR"
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

DIFFUSION_ARGS=""
if [[ "$BACKEND" == "diffusion" ]]; then
    DIFFUSION_ARGS="--mc_num ${MC_NUM} --mc_batch_size ${MC_BATCH_SIZE}"
fi

# Run the main evaluation
python -m evaluation_pipeline.AoA_word.run \
--model_name $MODEL_NAME \
--backend $BACKEND \
--track_name $TRACK_NAME \
--word_path $WORD_PATH \
--output_dir $OUTPUT_DIR \
--min_context $MIN_CONTEXT \
$DIFFUSION_ARGS

