from allennlp.data.batch import Batch
from allennlp.nn import util
import torch
from influence_info.influencers.base_influencer import BaseInfluencer
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
import numpy as np

@BaseInfluencer.register('representer_points')
class Representer_Points(BaseInfluencer):
    def __init__(self, predictor):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

        self._lambda = .001 #self._predictor._model._lambda

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs


    def compute_influence_values(self, training_loader, validation_loader):
        features = []
        logit_grads = []
        training_idx = []

        for batch in tqdm(iter(training_loader)):
            outputs = self.get_outputs_for_batch(batch)

            assert "loss" in outputs and "logits" in outputs and "features" in outputs, breakpoint()

            training_idx += outputs["idx"]
            features.append(outputs["features"].detach())

            probs = torch.nn.Softmax(dim=-1)(outputs["logits"])
            probs[torch.arange(probs.shape[0]), outputs['gold_labels']] -= 1

            logit_grads.append(probs.detach())

        features = torch.cat(features, dim=0).transpose(0, 1)  # (|T|, F)
        logit_grads = torch.cat(logit_grads, dim=0)
        alpha = logit_grads * (-1 / (2 * self._lambda * logit_grads.shape[0]))  # (|T|, L)

        similarity = []
        validation_idx = []
        original_prediction = []

        for batch_instances in tqdm(iter(validation_loader)):
            outputs = self.get_outputs_for_batch(batch_instances)

            assert "loss" in outputs and "logits" in outputs and "features" in outputs, breakpoint()
            val_features = outputs['features']

            validation_idx += outputs["idx"]
            similarity.append(torch.matmul(val_features, features)) #(B, |T|)

            original_prediction.append(outputs['logits'].detach())

        similarity = torch.cat(similarity, dim=0) #(|D|, |T|)

        influence_values = similarity.unsqueeze(-1) * alpha.unsqueeze(0) #(|D|, |T|, L)

        reconstructed_prediction = torch.nn.Softmax(dim=-1)(influence_values.sum(1)).cpu().data.numpy()
        original_prediction = torch.nn.Softmax(dim=-1)(torch.cat(original_prediction, dim=0)).cpu().data.numpy()

        print(np.corrcoef(original_prediction[:, 1], reconstructed_prediction[:, 1]))
        print(classification_report(original_prediction.argmax(-1), reconstructed_prediction.argmax(-1)))

        return influence_values.cpu().data.numpy(), training_idx, validation_idx





