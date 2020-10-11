set -eu

export CONFIG_FILE=influence_info/training_config/$CLASSIFIER.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE}

export TRAIN_DATA_PATH=$TRAIN_DATA_PATH
export DEV_DATA_PATH=$DEV_DATA_PATH
export TEST_DATA_PATH=$TEST_DATA_PATH

export OUTPUT_BASE_PATH=$OUTPUT_BASE_PATH/$CLASSIFIER/

export SEED=${RANDOM_SEED:-100}

if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" && -z "$again" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json does not exist ... . TRAINING ";
    allennlp train -s $OUTPUT_BASE_PATH --include-package influence_info --force $CONFIG_FILE
fi