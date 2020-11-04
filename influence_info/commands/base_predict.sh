set -eu

function predict {
    if [[ -f "$1"  && -z "$again" ]]; then 
        echo "$1 exists .. Not Predicting";
    else 
        echo "$1 do not exist ... Predicting";
        allennlp predict \
        --output-file $1 \
        --batch-size $BSIZE \
        --use-dataset-reader \
        --dataset-reader-choice validation \
        --predictor base_predictor \
        --include-package influence_info \
        --cuda-device $CUDA_DEVICE \
        --silent \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}

predict $OUTPUT_BASE_PATH/predictions.train.jsonl $TRAIN_DATA_PATH;
predict $OUTPUT_BASE_PATH/predictions.dev.jsonl $DEV_DATA_PATH;
predict $OUTPUT_BASE_PATH/predictions.test.jsonl $TEST_DATA_PATH;