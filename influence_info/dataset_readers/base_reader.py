import json
from typing import Any, Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ArrayField, LabelField, MetadataField,
                                  TextField)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides


@DatasetReader.register("base_reader")
class BaseReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], lazy: bool = False,) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for _, line in enumerate(data_file.readlines()):
                items = json.loads(line)
                document = items["document"]
                idx = items["idx"]
                query = items.get("query", None)
                label = items.get("label", None)

                if label is not None:
                    label = str(label).replace(" ", "_")

                instance = self.text_to_instance(idx=idx, document=document, query=query, label=label,)
                yield instance

    @overrides
    def text_to_instance(
        self, idx: str, document: str, query: str = None, label: str = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        document_tokens = self._tokenizer.tokenize(document)
        query_tokens = self._tokenizer.tokenize(query) if query is not None else []

        fields["tokens"] = TextField(query_tokens + document_tokens, token_indexers=self._token_indexers)

        metadata = {
            "idx": idx,
            "document": document,
            "label": label,
        }

        if query is not None:
            metadata["query"] = query

        fields["metadata"] = MetadataField(metadata)
        fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)
