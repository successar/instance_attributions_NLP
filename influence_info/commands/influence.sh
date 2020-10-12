set -eu

export TRAIN_DATA_PATH=$DATADIR/${DATASET_NAME}/data/train.jsonl
export DEV_DATA_PATH=$DATADIR/${DATASET_NAME}/data/dev.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME}/${EXP_NAME}/$CLASSIFIER/

export PYTHONPATH=.

python influence_info/influencers/compute_influence_values.py \
--archive-file $OUTPUT_BASE_PATH/model.tar.gz \
--training-file $TRAIN_DATA_PATH \
--validation-file $DEV_DATA_PATH \
--cuda-device $CUDA_DEVICE \
--training-batch-size ${BSIZE:-8} \
--validation-batch-size ${BSIZE:-8} \
--influencer-file influence_info/influencer_config/$INFLUENCER.jsonnet \
--output-folder $OUTPUT_BASE_PATH/$INFLUENCER/;