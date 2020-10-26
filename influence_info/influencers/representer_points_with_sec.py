import numpy as np
import torch
from allennlp.nn import util
from influence_info.influencers.base_influencer import BaseInfluencer
from scipy.special import softmax
from sklearn.metrics import classification_report
from tqdm import tqdm

from scipy.stats import pearsonr


class SupplementaryModel(torch.nn.Module):
    def __init__(self, model, feature_size, num_labels, reg=0.1):
        super().__init__()
        self.classifier = torch.nn.Linear(feature_size, num_labels)
        self.classifier.load_state_dict(model._classifier.state_dict())
        self.reg = reg

    def forward(self, X, y=None):
        logits = self.classifier(X)

        if y is None:
            return logits

        logsoft = torch.nn.LogSoftmax(dim=-1)(logits)

        loss = -(y * logsoft).sum(-1)
        loss = loss.mean()

        for p in self.parameters():
            loss += self.reg * (p ** 2).sum()

        return loss

    def optimize(self, X, y):
        optim = torch.optim.LBFGS(self.parameters(), max_iter=1000, history_size=500, line_search_fn="strong_wolfe")
        for _ in tqdm(range(3)):

            def closure():
                optim.zero_grad()
                loss = self.forward(X, y)
                loss.backward()
                return loss

            optim.step(closure)

    def predict_proba(self, features):
        return torch.nn.Softmax(dim=-1)(self.classifier(features))

    def predict(self, features):
        return self.classifier(features).argmax(-1)


@BaseInfluencer.register("representer_points_with_sec")
class Representer_Points_With_Sec(BaseInfluencer):
    def __init__(self, predictor):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs

    @classmethod 
    def run_all_configs(cls, predictor) :
        yield cls(predictor)

    def train_with_lbfgs(self, training_loader):
        features = []
        logits = []
        for batch in tqdm(iter(training_loader)):
            outputs = self.get_outputs_for_batch(batch)
            features.append(outputs["features"].detach())
            logits.append(outputs["logits"].detach())

        features = torch.cat(features, dim=0)
        probs = torch.nn.Softmax(dim=-1)(torch.cat(logits, dim=0))

        supp_model = SupplementaryModel(self._predictor._model, features.shape[1], probs.shape[1]).to(features.device)
        supp_model.optimize(features, probs)

        print(
            classification_report(
                probs.argmax(-1).cpu().data.numpy(), supp_model.predict(features).cpu().data.numpy()
            )
        )

        return supp_model

    def compute_influence_values(self, training_loader, validation_loader):
        print("Training LR model")
        lr_model = self.train_with_lbfgs(training_loader)
        features = []
        logit_grads = []
        training_idx = []

        for batch in tqdm(iter(training_loader)):
            with torch.no_grad() :
                outputs = self.get_outputs_for_batch(batch)

            assert "features" in outputs, breakpoint()

            training_idx += outputs["idx"]
            batch_features = outputs["features"].detach()
            features.append(batch_features.cpu().data.numpy())

            labels = outputs["gold_labels"].cpu().data.numpy()

            probs = lr_model.predict_proba(batch_features).detach().cpu().data.numpy()
            probs[np.arange(probs.shape[0]), labels] -= 1

            logit_grads.append(probs)

        features = np.concatenate(features, axis=0).T  # (E, |T|)
        logit_grads = np.concatenate(logit_grads, axis=0)
        alpha = -logit_grads * (1 / (2 * lr_model.reg * logit_grads.shape[0]))  # (|T|, L)

        similarity = []
        validation_idx = []
        original_prediction = []

        for batch_instances in tqdm(iter(validation_loader)):
            with torch.no_grad() :
                outputs = self.get_outputs_for_batch(batch_instances)

            assert "features" in outputs, breakpoint()
            val_features = outputs["features"].cpu().data.numpy()

            validation_idx += outputs["idx"]
            similarity.append((val_features @ features) + 1)  # (B, |T|)

            probs = torch.nn.Softmax(dim=-1)(outputs["logits"])

            original_prediction.append(probs.cpu().data.numpy())

        similarity = np.concatenate(similarity, axis=0)  # (|D|, |T|)

        influence_values = similarity[:, :, None] * alpha[None, :, :]  # (|D|, |T|, L)

        reconstructed_prediction = softmax(influence_values.sum(1), axis=1)
        original_prediction = np.concatenate(original_prediction, axis=0)

        corr = [
            pearsonr(original_prediction[i], reconstructed_prediction[i])[0]
            for i in range(original_prediction.shape[0])
        ]
        print(np.mean(corr))
        print(classification_report(original_prediction.argmax(-1), reconstructed_prediction.argmax(-1)))

        return influence_values, training_idx, validation_idx

