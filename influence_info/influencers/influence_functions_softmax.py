import torch
from allennlp.nn import util
from influence_info.influencers.base_influencer import BaseInfluencer
from tqdm import tqdm

norm = lambda x: torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9


class SuppModel:
    def __init__(self, classifier):
        self._classifier = classifier
        self._classifier.requires_grad = True
        self._has_bias = self._classifier.bias is not None
        weight, bias = self._classifier.weight, self._classifier.bias

        if self._has_bias:
            self._theta = torch.zeros((weight.shape[0], weight.shape[1] + 1))
            self._theta[:, :-1] = weight
            self._theta[:, -1] = bias

            self._theta = torch.tensor(self._theta, requires_grad=True, device=weight.device)
        else:
            self._theta = torch.tensor(weight, requires_grad=True, device=weight.device)

    def get_hessian(self, X, y):
        y = y.argmax(-1)

        def func(theta):
            logits = X @ theta.t()
            loss = torch.nn.CrossEntropyLoss(reduction="sum")(logits, y) + 0.001 * torch.sum(theta * theta)
            return loss

        hessian = torch.autograd.functional.hessian(func, self._theta)

        return hessian


@BaseInfluencer.register("influence_function_softmax")
class InfluenceFunctionExact(BaseInfluencer):
    def __init__(self, predictor, use_hessian: bool = True, normalize_grad: bool = False):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

        self._has_bias = self._predictor._model._classifier.bias is not None
        self._use_hessian = use_hessian
        self._normalise_grad = normalize_grad

    def get_output_subfolder(self):
        return f"use_hessian:{self._use_hessian},norm_grad:{self._normalise_grad}"

    @classmethod
    def run_all_configs(cls, predictor):
        for use_hessian in [True, False]:
            for normalize_grad in [True, False]:
                yield cls(predictor, use_hessian=use_hessian, normalize_grad=normalize_grad)

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs

    def get_pytorch_hessian(self, features, labels):
        # (T, F), (T, L), (T,)

        model = SuppModel(self._predictor._model._classifier)
        hessian = model.get_hessian(features, labels)
        return hessian

    def get_features_and_logits(self, dataloader):
        idx = []
        features = []
        logits = []
        labels = []
        for batch in tqdm(iter(dataloader)):
            outputs = self.get_outputs_for_batch(batch)
            idx += [m["idx"] for m in batch["metadata"]]
            features.append(outputs["features"].detach())
            logits.append(outputs["logits"].detach())
            labels += list(outputs["gold_labels"].cpu().data.numpy())

        features = torch.cat(features, dim=0)
        if self._has_bias:
            features = torch.cat([features, torch.ones((features.shape[0], 1)).to(features.device)], dim=-1)

        logits = torch.cat(logits, dim=0)
        labels = torch.eye(logits.shape[1])[labels].to(self._predictor.cuda_device)

        return idx, features, logits, labels

    def compute_influence_values(self, training_loader, validation_loader):
        (training_idx, training_features, training_logits, training_labels,) = self.get_features_and_logits(
            training_loader
        )
        (
            validation_idx,
            validation_features,
            validation_logits,
            validation_labels,
        ) = self.get_features_and_logits(validation_loader)

        fvals = training_features.sum(0) != 0.0
        training_features = training_features[:, fvals]
        validation_features = validation_features[:, fvals]

        training_probs = torch.nn.Softmax(dim=-1)(training_logits)
        validation_probs = torch.nn.Softmax(dim=-1)(validation_logits)

        feature_size, label_size = training_features.shape[1], training_probs.shape[1]

        if self._use_hessian:
            H = 0
            H_pred = torch.diag_embed(training_probs) - torch.bmm(
                training_probs.unsqueeze(-1), training_probs.unsqueeze(1)
            )  # (T, L, L)

            for f, h_pred in tqdm(zip(training_features, H_pred)):
                f = f.unsqueeze(1)
                h_feat = f @ f.t()
                H += h_feat.unsqueeze(0).unsqueeze(2) * h_pred.unsqueeze(1).unsqueeze(3)

            H = H.reshape(feature_size * label_size, feature_size * label_size) / training_features.shape[0]
            H += torch.eye(H.shape[0]).to(H.device) * 0.001

            assert torch.allclose(H, H.t()), breakpoint()

            try:
                H_inv = torch.inverse(H)
            except:
                breakpoint()
        else:
            H_inv = torch.eye(feature_size * label_size).to(training_features.device)

        training_grad = (training_labels - training_probs).unsqueeze(1) * training_features.unsqueeze(-1)
        training_grad = training_grad.reshape(training_grad.shape[0], -1)  # (T, G)

        influence_values_list = []
        for i in range(validation_labels.shape[-1]) :
            validation_labels = torch.zeros_like(validation_labels)
            validation_labels[:, i] = 1.0
            validation_grad = (validation_labels - validation_probs).unsqueeze(1) 
            validation_grad = validation_grad * validation_features.unsqueeze(-1)
            validation_grad = validation_grad.reshape(validation_grad.shape[0], -1)  # (D, G)
            influence_values = -validation_grad @ (training_grad @ H_inv).t()  # (D, T)

            if self._normalise_grad:
                training_norm = norm(training_grad @ H_inv)  # (T, 1)
                validation_norm = norm(validation_grad @ H_inv)  # (D, 1)
                norm_matrix = validation_norm @ training_norm.t()  # (D, T)

                influence_values = influence_values / norm_matrix

            influence_values_list.append(influence_values.unsqueeze(-1))

        influence_values = torch.cat(influence_values_list, dim=-1)

        return influence_values.cpu().data.numpy(), training_idx, validation_idx

