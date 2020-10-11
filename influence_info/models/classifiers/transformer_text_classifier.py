from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy

import torch.nn as nn

from typing import Optional, Dict


@Model.register("transformer_text_classifier")
class TransformerTextClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super().__init__(vocab, regularizer)
        self._vocab = vocab
        self._text_field_embedder = text_field_embedder
        self._pooler = seq2vec_encoder

        self._num_labels = self._vocab.get_vocab_size("labels")
        self._classifier = nn.Linear(self._pooler.get_output_dim(), self._num_labels, bias=False)

        self._f1 = FBetaMeasure()
        self._accuracy = CategoricalAccuracy()

        initializer(self)

    def forward(self, tokens, label, metadata):
        embeddings = self._text_field_embedder(tokens)
        pooled_output = self._pooler(embeddings)

        logits = self._classifier(pooled_output)

        output_dict = {}
        output_dict["logits"] = logits
        output_dict["features"] = pooled_output
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
        output_labels = self._vocab.get_index_to_token_vocabulary("labels")
        output_labels = [output_labels[i] for i in range(len(output_labels))]

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

