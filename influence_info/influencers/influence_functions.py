import re

import numpy as np
import torch
import torch.autograd as autograd
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.nn import util as util
from influence_info.influencers.base_influencer import BaseInfluencer
from tqdm import tqdm


@BaseInfluencer.register("influence_function")
class InfluenceFunctions(BaseInfluencer):
    def __init__(self, predictor, param_regex: str, use_hessian: bool = True):
        self._predictor = predictor
        self._predictor._model.eval()

        print("Parameters requiring grad are : ")
        for name, p in self._predictor._model.named_parameters():
            if re.search(param_regex, name) is None:
                p.requires_grad = False
            else:
                print(name)
                p.requires_grad = True

        self._valid_parameters = tuple(
            p for p in self._predictor._model.parameters() if p.requires_grad == True
        )

        self._damping = 0.01
        self._scale = 25
        self._use_hessian = use_hessian

    def get_output_subfolder(self) :
        return f"use_hessian:{self._use_hessian}"

    def compute_influence_values(
        self, training_loader: PyTorchDataLoader, validation_loader: PyTorchDataLoader
    ):
        influence_values = []
        validation_idx = []

        for batch in tqdm(iter(validation_loader)):
            assert len(batch["metadata"]) == 1, breakpoint()
            influence_values.append([])
            ihvp = self.ihvp(batch, training_loader)  # (tuple of params)
            validation_idx.append(batch["metadata"][0]["idx"])

            training_idx = []
            for train_ex in iter(training_loader):
                assert len(train_ex["metadata"]) == 1, breakpoint()
                train_grad = self.get_grad(train_ex)

                if_value = -sum((x * y).sum().item() for x, y in zip(ihvp, train_grad)) / len(training_loader)
                influence_values[-1].append(if_value)

                training_idx.append(train_ex["metadata"][0]["idx"])

        return np.array(influence_values), training_idx, validation_idx

    def get_outputs_for_batch(self, batch):
        cuda_device = self._predictor.cuda_device
        model_input = util.move_to_device(batch, cuda_device)
        outputs = self._predictor._model(**model_input)

        return outputs["loss"]

    def get_grad(self, batch):
        loss = self.get_outputs_for_batch(batch)
        grads = autograd.grad(loss, self._valid_parameters)

        return grads

    def ihvp(self, test_example, training_loader):
        self._predictor._model.zero_grad()
        v = self.get_grad(test_example)

        if not self._use_hessian:
            return tuple(x.detach() for x in v)

        ihv_estimate = v

        training_loader = PyTorchDataLoader(training_loader.dataset, batch_size=5, shuffle=True)
        training_iter = iter(training_loader)
        for _ in range(len(training_loader)):
            train_batch = next(training_iter)

            self._predictor._model.zero_grad()

            loss = self.get_outputs_for_batch(train_batch)
            hv = vhp_s(loss, self._valid_parameters, ihv_estimate)

            with torch.no_grad():
                ihv_estimate = tuple(
                    _v + (1 - self._damping) * _ihv - _hv / self._scale
                    for _v, _ihv, _hv in zip(v, ihv_estimate, hv)
                )

        return tuple(x.detach() for x in ihv_estimate)


def vhp_s(loss, model_params, v):  # according to pytorch issue #24004
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv
