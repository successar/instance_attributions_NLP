import json
import os
from typing import Iterable

import numpy as np
from allennlp.common.params import Params
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.models import load_archive
from allennlp.predictors import Predictor
from influence_info.influencers.base_influencer import BaseInfluencer


def read_data(reader: DatasetReader, file: str) -> Iterable[Instance]:
    print(f"Reading data from {file}")
    training_data = reader.read(file)
    return training_data


def get_predictor(args):
    print(f"Loading Model from {args.archive_file}")
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device,)

    return Predictor.from_archive(archive, predictor_name="base_predictor")


def get_influencer(predictor, args):
    params = Params.from_file(args.influencer_file)
    influencer = BaseInfluencer.from_params(params, predictor=predictor)
    return influencer


def dump_results(influence_values, training_idx, validation_idx, args):
    os.makedirs(args.output_folder, exist_ok=True)
    np.save(os.path.join(args.output_folder, "influence_values.npy"), influence_values)
    json.dump(training_idx, open(os.path.join(args.output_folder, "training_idx.json"), "w"))
    json.dump(validation_idx, open(os.path.join(args.output_folder, "validation_idx.json"), "w"))


def run(args):
    predictor: Predictor = get_predictor(args)

    training_file = args.training_file
    validation_file = args.validation_file

    training_data = read_data(predictor._dataset_reader, training_file)
    validation_data = read_data(predictor._dataset_reader, validation_file)

    print("Indexing with Vocabulary")
    training_data.index_with(predictor._model.vocab)
    validation_data.index_with(predictor._model.vocab)

    training_loader = PyTorchDataLoader(training_data, batch_size=args.training_batch_size, shuffle=False)
    validation_loader = PyTorchDataLoader(
        validation_data, batch_size=args.validation_batch_size, shuffle=False
    )

    print("Computing Influence Values")
    influencer = get_influencer(predictor, args)
    influence_values, training_idx, validation_idx = influencer.compute_influence_values(
        training_loader, validation_loader
    )

    dump_results(influence_values, training_idx, validation_idx, args)

    print("Job done. Rejoice !")


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--archive-file")
parser.add_argument("--influencer-file")
parser.add_argument("--cuda-device", type=int)
parser.add_argument("--training-file")
parser.add_argument("--validation-file")
parser.add_argument("--training-batch-size", type=int)
parser.add_argument("--validation-batch-size", type=int)
parser.add_argument("--output-folder")

if __name__ == "__main__":
    from allennlp.common.util import import_module_and_submodules

    import_module_and_submodules("influence_info")
    args = parser.parse_args()

    run(args)

