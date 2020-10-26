from allennlp.common.registrable import Registrable
from typing import Iterable

class BaseInfluencer(Registrable) :
    def __init__(self) :
        pass

    def compute_influence_values(self, training_loader, validation_loader) :
        raise NotImplementedError

    def get_output_subfolder(self) :
        return ""

    @classmethod
    def run_all_configs(cls, predictor) -> Iterable:
        raise NotImplementedError