import numpy as np
from allennlp.nn import util
from influence_info.influencers.base_influencer import BaseInfluencer
from tqdm import tqdm

norm = lambda x : x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

sim_func_dict = {
    "cos" : lambda x, y: norm(x) @ norm(y).T,
    "dot" : lambda x, y: x @ y.T,
    "euc" : lambda x, y: -np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
}


@BaseInfluencer.register("nearest_neighbors")
class NearestNeighbors(BaseInfluencer):
    def __init__(self, predictor, similarity_function: str):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

        self._sim_func = similarity_function
        self._similarity_function = sim_func_dict[similarity_function]

    def get_output_subfolder(self) :
        return f"sim_func:{self._sim_func}"

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs

    def compute_influence_values(self, training_loader, validation_loader):
        features = []
        training_idx = []

        for batch in tqdm(iter(training_loader)):
            outputs = self.get_outputs_for_batch(batch)

            assert "features" in outputs, breakpoint()

            training_idx += outputs["idx"]
            batch_features = outputs["features"].detach()
            features.append(batch_features.cpu().data.numpy())

        features = np.concatenate(features, axis=0)

        similarity = []
        validation_idx = []

        for batch_instances in tqdm(iter(validation_loader)):
            outputs = self.get_outputs_for_batch(batch_instances)

            assert "features" in outputs, breakpoint()
            val_features = outputs["features"].cpu().data.numpy()

            validation_idx += outputs["idx"]
            similarity.append(self._similarity_function(val_features, features))  # (B, |T|)

        similarity = np.concatenate(similarity, axis=0)  # (|D|, |T|)

        return similarity, training_idx, validation_idx

