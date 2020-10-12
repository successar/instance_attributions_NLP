{
  dataset_reader : {
    type : "image_reader",
  },
  train_data_path: 'train',
  validation_data_path: 'dev',
  test_data_path: 'test',
  model: {
    type: "image_classifier",
  },
  data_loader : {
    batch_size: std.parseInt(std.extVar('BSIZE')),
    shuffle: true,
  },
  trainer: {
    num_epochs: std.parseInt(std.extVar('EPOCHS')),
    patience: 30,
    grad_norm: 5.0,
    validation_metric: "+validation_metric",
    checkpointer: {num_serialized_models_to_keep: 1,},
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    optimizer: {
      type: "adam",
      lr: 1e-3
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}