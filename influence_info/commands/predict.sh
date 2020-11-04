set -eu

export TRAIN_DATA_PATH=${DATADIR:-Datasets}/${DATASET_NAME}/data/train.jsonl
export DEV_DATA_PATH=${DATADIR:-Datasets}/${DATASET_NAME}/data/dev.jsonl
export TEST_DATA_PATH=${DATADIR:-Datasets}/${DATASET_NAME}/data/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME}/${EXP_NAME}/${CLASSIFIER}

bash influence_info/commands/base_predict.sh