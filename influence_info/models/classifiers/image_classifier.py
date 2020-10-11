from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


@Model.register("image_classifier")
class ImageClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super().__init__(vocab, regularizer)
        self._vocab = vocab
        self._num_labels = 10

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self._classifier = nn.Linear(128, self._num_labels)

        self._f1 = FBetaMeasure()
        self._accuracy = CategoricalAccuracy()

        initializer(self)

    def forward(self, image, label, metadata):
        x = self.conv1(image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self._classifier(x)

        output_dict = {}
        output_dict["logits"] = logits
        output_dict["features"] = x 
        output_dict["metadata"] = metadata
        output_dict["idx"] = [m["idx"] for m in metadata]

        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)
            output_dict["loss"] = loss
            output_dict["gold_labels"] = label
            self._call_metrics(output_dict)

        return output_dict

    def _call_metrics(self, output_dict):
        self._f1(output_dict["logits"], output_dict["gold_labels"])
        self._accuracy(output_dict["logits"], output_dict["gold_labels"])

    def make_output_human_readable(self, output_dict):
        result_output_dict = {}
        result_output_dict["logits"] = output_dict["logits"].cpu().data.numpy()
        result_output_dict["predicted_labels"] = output_dict["logits"].cpu().data.numpy().argmax(-1)
        result_output_dict["idx"] = output_dict["idx"]
        result_output_dict["gold_labels"] = output_dict["gold_labels"].cpu().data.numpy()

        return result_output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._f1.get_metric(reset)
        macro_avg = {"macro_" + k: sum(v) / len(v) for k, v in metrics.items()}
        output_labels = list(range(self._num_labels))

        class_metrics = {}
        for k, v in metrics.items():
            assert len(v) == len(output_labels)
            class_nums = dict(zip(output_labels, v))
            class_metrics.update({k + "_" + str(kc): x for kc, x in class_nums.items()})

        class_metrics.update({"accuracy": self._accuracy.get_metric(reset)})
        class_metrics.update(macro_avg)
        modified_class_metrics = {}

        for k, v in class_metrics.items():
            if k in ["accuracy", "macro_fscore"]:
                modified_class_metrics[k] = v
            else:
                modified_class_metrics["_" + k] = v

        modified_class_metrics["validation_metric"] = class_metrics["macro_fscore"]

        return modified_class_metrics

