import inspect
from functools import partial
from typing import Callable, Optional

import numpy as np
from art.estimators.classification import PyTorchClassifier
from torch import Tensor, from_numpy, nn
from numpy import ndarray

def art_wrapper(attack: Callable,
                model: nn.Module,
                inputs: Tensor,
                labels: Tensor,
                targets: Optional[Tensor] = None,
                targeted: bool = False,
                target_model: Optional[nn.Module] = None) -> Tensor:
    loss = nn.CrossEntropyLoss()
    input_shape = inputs.shape[1:]
    output_shape = [module for module in model.modules()][-1].out_features

    art_model = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss, input_shape=input_shape,
                                  nb_classes=output_shape)
    model_arg = next(iter(inspect.signature(attack.func).parameters))
    attack_kwargs = {model_arg: art_model}

    if 'targeted' in attack.func.attack_params:  # not all attacks have the targeted arg
        attack_kwargs['targeted'] = targeted

    attack = attack(batch_size=len(inputs), **attack_kwargs)
    y = targets if targeted else labels
    if target_model == None:
        adv_inputs = attack.generate(x=inputs.cpu().numpy(), y=y.cpu().numpy())
    else:
        ''''t_art_model = PyTorchClassifier(model=target_model, clip_values=(0, 1), loss=loss, input_shape=input_shape,
                                  nb_classes=output_shape)
        t_attack_kwargs = {model_arg: t_art_model}

        if 'targeted' in attack.func.attack_params:  # not all attacks have the targeted arg
            t_attack_kwargs['targeted'] = targeted

        t_attack = attack(batch_size=len(inputs), **t_attack_kwargs)'''
        
        adv_inputs = attack.generate(x=inputs.cpu().numpy(), y=y.cpu().numpy(), target_model = target_model)

    return from_numpy(adv_inputs).to(inputs.device)


class ArtMinimalWrapper:
    def __init__(self, attack: partial, init_eps: float, max_eps: Optional[float] = None, search_steps: int = 10,
                 batched: bool = False):
        self.attack_partial = attack
        self.init_eps = init_eps
        self.max_eps = max_eps
        self.search_steps = search_steps
        self.batched = batched

    @property
    def func(self):
        return self.attack_partial.func

    def __call__(self, **kwargs):
        self.attack_kwargs = kwargs
        return self

    def generate(self, x: np.ndarray, y: np.ndarray, target_model: Optional[nn.Module]=None) -> np.ndarray:
        #x_tensor = from_numpy(x)
        batch_size = len(x)
        batch_view = lambda ndarray: ndarray.reshape(batch_size, *[1] * (x.ndim - 1))

        adv_inputs = x.copy()
        eps_low = np.full_like(y, 0, dtype=x.dtype)
        best_eps = np.full_like(eps_low, float('inf') if self.max_eps is None else self.max_eps)
        found_high = np.full_like(eps_low, False, dtype=bool)

        eps = np.full_like(eps_low, self.init_eps)
        for i in range(self.search_steps):
            if self.batched:
                eps_step = np.full_like(eps, self.attack_partial.keywords['eps_step'])
                #attack = self.attack_partial(eps=batch_view(eps).cpu().numpy(), **self.attack_kwargs)
                attack = self.attack_partial(eps=batch_view(eps), eps_step=batch_view(eps_step), **self.attack_kwargs)
                adv_inputs_run = attack.generate(x=x, y=y)
            else:
                # run attacks by batching based on unique epsilons
                adv_inputs_run = x.copy()
                unique_eps = np.unique(eps)
                for eps_ in unique_eps:
                    mask = eps == eps_
                    attack = self.attack_partial(eps=float(eps_), **self.attack_kwargs)
                    adv_inputs_run[mask] = attack.generate(x=x[mask], y=y[mask])

            if target_model == None:     
                logits = attack.estimator.predict(x=adv_inputs_run, batch_size=len(adv_inputs_run))
            else:
                #logits = target_attack.estimator.predict(x=adv_inputs_run, batch_size=len(adv_inputs_run))
                logits = target_model(from_numpy(adv_inputs_run).to(next(target_model.parameters()).device)).cpu().detach().numpy()

            preds = np.argmax(logits, axis=1)
            is_adv = (preds == y) if attack.targeted else (preds != y)

            better_adv = is_adv & (eps < best_eps)
            adv_inputs[better_adv] = adv_inputs_run[better_adv]

            found_high |= better_adv
            eps_low = np.where(better_adv, eps_low, eps)
            best_eps = np.where(better_adv, eps, best_eps)

            eps = np.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

        return adv_inputs
