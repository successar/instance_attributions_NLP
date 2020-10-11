from allennlp.common.registrable import Registrable

class BaseInfluencer(Registrable) :
    def __init__(self) :
        pass

    def compute_influence_values(self, training_loader, validation_loader) :
        raise NotImplementedError