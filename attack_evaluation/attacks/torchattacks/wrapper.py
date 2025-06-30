from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from torchattacks.attack import Attack


def torchattacks_wrapper(attack: Attack,
                         model: nn.Module,
                         inputs: Tensor,
                         labels: Tensor,
                         targets: Optional[Tensor] = None,
                         targeted: bool = False,
                         target_model: Optional[nn.Module] = None) -> Tensor:
    attack = attack(model=model)
    if targeted:
        attack.set_mode_targeted_by_function()
    if target_model == None:
        adv_examples = attack(inputs, labels=targets if targeted else labels)
    else:
        adv_examples = attack(inputs, labels=targets if targeted else labels, target_model = target_model)
    return adv_examples


class TorchattacksMinimalWrapper:
    def __init__(self, model: nn.Module, attack: partial, init_eps: float, search_steps: int,
                 max_eps: Optional[float] = None, batched: bool = False):
        self.attack = attack(model=model)
        self.init_eps = init_eps
        self.search_steps = search_steps
        self.max_eps = max_eps
        self.batched = batched
        self.targeted = False

    def set_mode_targeted_by_function(self):
        self.attack.set_mode_targeted_by_function()
        self.targeted = True

    def __call__(self, inputs: Tensor, labels: Tensor, target_model: Optional[nn.Module] = None) -> Tensor:
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        adv_inputs = inputs.clone()
        eps_low = inputs.new_zeros(batch_size)
        best_eps = torch.full_like(eps_low, float('inf') if self.max_eps is None else self.max_eps)
        found_high = torch.full_like(eps_low, False, dtype=torch.bool)

        eps = torch.full_like(eps_low, self.init_eps)
        for i in range(self.search_steps):
            if self.batched:
                #self.attack.eps = batch_view(eps)##for pgd-l2
                adv_inputs_run = self.attack(inputs, labels=labels)
            else:
                adv_inputs_run = inputs.clone()
                for eps_ in torch.unique(eps):
                    mask = eps == eps_
                    self.attack.eps = eps_.item()
                    adv_inputs_run[mask] = self.attack(inputs[mask], labels=labels[mask])

            if target_model == None:
                logits = self.attack.model(adv_inputs_run)
            else:
                logits = target_model(adv_inputs_run)
            preds = logits.argmax(dim=1)
            is_adv = (preds == labels) if self.targeted else (preds != labels)

            better_adv = is_adv & (eps < best_eps)
            adv_inputs[better_adv] = adv_inputs_run[better_adv]

            found_high.logical_or_(better_adv)
            eps_low = torch.where(better_adv, eps_low, eps)
            best_eps = torch.where(better_adv, eps, best_eps)

            eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

        return adv_inputs
