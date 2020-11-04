This project uses Allennlp + Pytorch.

To install the requirements, use the `conda_env.yml` file .

I am storing data in the `Datasets/` folder (you can also use some other folder but you need to set a `DATADIR` environment variable ). For any dataset (for example, SST), the data should be available in following dirs -

1. `Datasets/{dataset_name}/data/{train,dev,test.jsonl}` (or `{DATADIR}/{dataset_name}/data/{train,dev,test.jsonl}`)
2. Each line in jsonl file is of form 

```json
{
    "idx" : str,
    "document": str,
    "query": Optional[str],
    "label": str
}
```

For NLI, we could generate dataset in above format by setting for each example, the document keys as `premise [SEP] hypothesis` .

The code is stored in the `influence_info` dir. The structure is as follows :

1. dataset_readers - Contains dataset reader (see Allennlp) for use in our models. The reader for text data is `base_reader.py` . Code should be self explanatory.
2. models - Contains models we want to train in `classifiers` subfolder. For text data, we want to use the `transformer_text_classifier.py` . This basically uses a BERT model with a linear classifier on top. We use the pretrained transformer embedder from Allennlp.
3. training_config - Contains the jsonnet configuration files for training models. For text data, we want to use the `transformer_text_classifier.jsonnet`. Some configurations can be set by using environment variables. See `std.extVar` statements in the file. Should be self explanatory.

4. Most of my code is run using bash scripts in the `commands` folder.

    1. For training a model, the file to use is `train.sh` (which calls `base_train.sh` after setting some paths). It takes a few environment variables,

    ```bash
    DATADIR=Datasets \
    OUTPUT_DIR=outputs \
    CUDA_DEVICE=0 \
    DATASET_NAME=SST \
    CLASSIFIER=transformer_text_classifier \
    EPOCHS=10 \
    BSIZE=8 \
    EXP_NAME=<your-experiment-name> \
    bash influence_info/commands/train.sh
    ```

    This will store your model in `{OUTPUT_DIR}/{DATASET_NAME}/{EXP_NAME}/{CLASSIFIER}` .

    Note this code also makes predictions on your train,dev,test files and store them the same output folder as `predictions.{train,dev,test}.jsonl` files (see `base_predict.sh`)

5. One you have a trained model, you can run all attribution methods (and all their subtypes) on it by just using the `influence_all.sh` file in commands folder.

    ```bash
    BSIZE=8 \
    EXP_NAME=<your-experiment-name> \
    DATASET_NAME=SST \
    DATADIR=Datasets \
    CUDA_DEVICE=0 \
    OUTPUT_DIR=outputs \
    CLASSIFIER=transformer_text_classifier \
    bash influence_info/commands/influence_all.sh
    ```

    For each influence method type, the results are stored in folder `{OUTPUT_DIR}/{DATASET_NAME}/{EXP_NAME}/{CLASSIFIER}\{name of the influencer}\{influencer configuration}\`

    This contains 3 file -

    1. influence_values.npy - This is a numpy matrix that can be loaded using np.load. It is a matrix of size (Validation set size, Training set size, number of classes) .
    2. training_idx.json - Maps the index of each element in the second dimension above to idx field in training data file.
    3. validation_idx.json - Maps the index of each element in the first dimension above to idx field in validation data file.






