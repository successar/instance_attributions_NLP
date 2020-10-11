from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

class BaseTextClassifier(Model) :
    def __init__(self, vocab: Vocabulary) :
        self._vocab = vocab

    def forward(self, document, query, label, metadata) :
        raise NotImplementedError

    def make_output_human_readable(self, output_dict) :
        raise NotImplementedError

    
    