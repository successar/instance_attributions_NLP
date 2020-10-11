local bert_type = std.extVar('BERT_MODEL_NAME');

{
  dataset_reader : {
    type : "base_reader",
    tokenizer: {
      type: "pretrained_transformer",
      model_name: bert_type,
      max_length: 512
    },
    token_indexers : {
      tokens: {
        type:"pretrained_transformer",
        model_name: bert_type,
        namespace: "transformer_tags",
        max_length: 512
      },
    }
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "transformer_text_classifier",
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: "pretrained_transformer",
          model_name: bert_type,
          max_length: 512,
          train_parameters: false
        },
      },
    },
    seq2vec_encoder : {
      type: "bert_pooler",
      pretrained_model: bert_type,
    },
    regularizer: {
      regexes: [["_classifier.*", {type : "l2", alpha: 0.1}]]
    }
  },
  data_loader : {
    batch_size: std.parseInt(std.extVar('BSIZE')),
    shuffle: true,
  },
  trainer: {
    num_epochs: std.parseInt(std.extVar('EPOCHS')),
    patience: 10,
    grad_norm: 5.0,
    validation_metric: "+validation_metric",
    checkpointer: {num_serialized_models_to_keep: 1,},
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    optimizer: {
      type: "adamw",
      lr: 2e-3
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}