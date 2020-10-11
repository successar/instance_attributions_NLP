from typing import Any, Dict, List, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, MetadataField
from allennlp.data.instance import Instance
from overrides import overrides

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import os


@DatasetReader.register("image_reader")
class ImageReader(DatasetReader):
    def __init__(self, lazy: bool = False, as_is=False) -> None:
        super().__init__(lazy=lazy)
        self._as_is = as_is

    @overrides
    def _read(self, file_path):
        data_base_dir = os.environ.get("DATADIR", "Datasets/")
        mnist_data = MNIST(root=data_base_dir, download=True, train="train" in file_path, transform=ToTensor())
        done_point = 5000 if "train" in file_path else 500
        for i, b in enumerate(mnist_data):
            image, label = b[0].numpy(), b[1]
            if i == done_point :
                break

            if self._as_is :
                yield (i, image, label)
                
            yield self.text_to_instance(i, image, label)

    @overrides
    def text_to_instance(self, idx, image, label) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        fields["image"] = ArrayField(image)

        metadata = {
            "idx": idx,
            "label": label,
        }

        fields["metadata"] = MetadataField(metadata)
        fields["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

        return Instance(fields)
