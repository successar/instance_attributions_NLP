import numpy as np
from allennlp.nn import util
from influence_info.influencers.base_influencer import BaseInfluencer
from scipy.special import softmax
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm


@BaseInfluencer.register("representer_points_with_LR")
class Representer_Points_With_LR(BaseInfluencer):
    def __init__(self, predictor):
        self._predictor = predictor
        self._predictor._model.eval()
        self._predictor._model.requires_grad = False

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs

    def train_with_lbfgs(self, training_loader):
        features = []
        labels = []
        for batch in tqdm(iter(training_loader)):
            outputs = self.get_outputs_for_batch(batch)
            features.append(outputs["features"].detach().cpu().data.numpy())
            labels.append(outputs["gold_labels"].cpu().data.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        lr = LogisticRegressionCV(penalty="l2")
        lr.fit(features, labels)

        print(classification_report(labels, lr.predict(features)))

        return lr

    def compute_influence_values(self, training_loader, validation_loader):
        print("Training LR model")
        lr_model = self.train_with_lbfgs(training_loader)
        features = []
        logit_grads = []
        training_idx = []

        for batch in tqdm(iter(training_loader)):
            outputs = self.get_outputs_for_batch(batch)

            assert "features" in outputs, breakpoint()

            training_idx += outputs["idx"]
            batch_features = outputs["features"].detach().cpu().data.numpy()
            features.append(batch_features)

            labels = outputs["gold_labels"].cpu().data.numpy()

            probs = lr_model.predict_proba(batch_features)
            probs[np.arange(probs.shape[0]), labels] -= 1

            logit_grads.append(probs)

        features = np.concatenate(features, axis=0).T  # (E, |T|)
        logit_grads = np.concatenate(logit_grads, axis=0)
        alpha = -logit_grads * lr_model.C_[0]  # (|T|, L)

        similarity = []
        validation_idx = []
        original_prediction = []

        for batch_instances in tqdm(iter(validation_loader)):
            outputs = self.get_outputs_for_batch(batch_instances)

            assert "features" in outputs, breakpoint()
            val_features = outputs["features"].cpu().data.numpy()

            validation_idx += outputs["idx"]
            similarity.append(val_features @ features)  # (B, |T|)

            original_prediction.append(lr_model.predict_proba(val_features))

        similarity = np.concatenate(similarity, axis=0)  # (|D|, |T|)

        influence_values = similarity[:, :, None] * alpha[None, :, :]  # (|D|, |T|, L)

        reconstructed_prediction = softmax(influence_values.sum(1), axis=1)
        original_prediction = np.concatenate(original_prediction, axis=0)

        corr = [np.corrcoef(original_prediction[i], reconstructed_prediction[i])[0, 1] for i in range(original_prediction.shape[0])]
        print(np.mean(corr))
        print(classification_report(original_prediction.argmax(-1), reconstructed_prediction.argmax(-1)))

        return influence_values, training_idx, validation_idx

