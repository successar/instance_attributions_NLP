import torch
from allennlp.nn import util
from influence_info.influencers.base_influencer import BaseInfluencer
from tqdm import tqdm

norm = lambda x : x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9)


@BaseInfluencer.register("influence_function_softmax")
class InfluenceFunctionExact(BaseInfluencer):
    def __init__(self, predictor):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs

    def get_features_and_logits(self, dataloader):
        idx = []
        features = []
        logits = []
        labels = []
        for batch in tqdm(iter(dataloader)):
            outputs = self.get_outputs_for_batch(batch)
            idx += [m['idx'] for m in batch['metadata']]
            features.append(outputs["features"].detach())
            logits.append(outputs["logits"].detach())
            labels += list(outputs["gold_labels"].cpu().data.numpy())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.eye(logits.shape[1])[labels].to(self._predictor.cuda_device)

        return idx, features, logits, labels

    def compute_influence_values(self, training_loader, validation_loader):
        (
            training_idx,
            training_features,
            training_logits,
            training_labels,
        ) = self.get_features_and_logits(training_loader)
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

        H = 0
        H_pred = torch.diag_embed(training_probs) - torch.bmm(
            training_probs.unsqueeze(-1), training_probs.unsqueeze(1)
        )  # (T, L, L)

        for f, h_pred in tqdm(zip(training_features, H_pred)):
            h_feat = f.unsqueeze(-1) * f.unsqueeze(0)  # (E, E)
            H += h_feat.unsqueeze(-1).unsqueeze(-1) * h_pred.unsqueeze(0).unsqueeze(0)

        H = H.transpose(1, 2)
        feature_size, label_size = H.shape[0], H.shape[1]
        H = H.reshape(feature_size * label_size, feature_size * label_size)

        try :
            H_inv = torch.inverse(H)
        except :
            breakpoint()

        training_grad = (training_labels - training_probs).unsqueeze(
            1
        ) * training_features.unsqueeze(-1)
        training_grad = training_grad.reshape(training_grad.shape[0], -1)

        H_inv_training_grad = torch.matmul(training_grad, H_inv)

        validation_grad = (validation_labels - validation_probs).unsqueeze(
            1
        ) * validation_features.unsqueeze(-1)
        validation_grad = validation_grad.reshape(validation_grad.shape[0], -1)

        influence_values = -norm(validation_grad) @ norm(training_grad).transpose(0, 1)

        return influence_values.cpu().data.numpy(), training_idx, validation_idx

