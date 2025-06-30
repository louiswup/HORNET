import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable
import scipy.stats as st
from functools import partial
from typing import Optional
from torch import Tensor

#############
# utils
#############

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_kernel(kernel_size=7):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3).transpose(2, 3, 0, 1)
    stack_kernel = torch.from_numpy(stack_kernel).cuda()
    return stack_kernel


# attack
def GA_TDMI_fgsm(model, eval_model, x_nature, y, max_eps, intervals, 
                 num_steps, momental, thres, kernel_size, loss_fn):
    batch_size = x_nature.shape[0]
    g = torch.zeros_like(x_nature)
    delta = torch.zeros_like(x_nature).cuda()
    eps_list = [(idx + 1) * (max_eps / intervals) for idx in range(intervals)]
    mask = torch.ones((batch_size, )).bool()
    for eps in eps_list:
        eps /= 255.0
        step_size = 1.25 * eps / num_steps
        delta = Variable(delta.data, requires_grad=True)

        for _ in range(num_steps):
            delta.requires_grad_()
            adv = x_nature + delta
            adv = torch.clamp(adv, 0, 1)
            with torch.enable_grad():
                #ensem_logits = model(adv, diversity=True)
                ensem_logits = model(input_diversity(adv,adv.shape[2]))
                loss = loss_fn(ensem_logits, y)

            PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
            PGD_grad = F.conv2d(PGD_grad, weight=get_kernel(kernel_size), stride=(
                1, 1), groups=3, padding=(kernel_size - 1) // 2)
            PGD_noise = PGD_grad / \
                torch.abs(PGD_grad).mean(dim=(1, 2, 3), keepdim=True)
            g[mask] = g[mask] * momental + PGD_noise[mask]
            delta = Variable(delta.data + step_size *
                             torch.sign(g), requires_grad=True)
            delta = Variable(torch.clamp(
                delta.data, -eps, eps), requires_grad=True)

        g.zero_()
        with torch.no_grad():
            tmp = x_nature + delta
            tmp = torch.clamp(tmp, 0, 1)
            #output = eval_model(tmp, diversity=False).detach()
            output = eval_model(tmp).detach()
        prob = F.softmax(output, dim=1)
        conf = prob[np.arange(batch_size), y.long()]
        mask = (conf >= thres)

        # early stopping
        if mask.sum() == 0:
            break

    X_pgd = Variable(x_nature + delta, requires_grad=False)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
    #budget = torch.abs(X_pgd - x_nature).reshape(batch_size, -1).max(dim = -1)[0]
    return X_pgd#, budget

def ga_tdmi_fgsm_wrapper(model: nn.Module,
                         inputs: Tensor,
                         labels: Tensor,
                         targets: Optional[Tensor] = None,
                         targeted: bool = False,
                         max_eps: float = 20,
                         intervals: int = 5,
                         num_steps: int = 10,
                         momental: float = 1.0,
                         thres: float = 0.2,
                         kernel_size: int = 5,
                         loss_fn: str = 'ce',
                         target_model: Optional[nn.Module] = None) -> Tensor:
    # loss_fn
    if loss_fn == "ce":
        loss = CrossEntropyLoss()
    elif loss_fn == "margin":
        loss = MarginLoss()
    else:
        raise Exception("invalid loss function!")
    #target model
    if target_model == None:
        target_model = model
    return GA_TDMI_fgsm(model=model, eval_model=target_model, x_nature=inputs, y=labels, max_eps=max_eps, intervals=intervals,
                 num_steps=num_steps, momental=momental, thres=thres, kernel_size=kernel_size, loss_fn=loss)

def GA_DMI_fgsm(model, eval_model, x_nature, y, max_eps, intervals, 
                 num_steps, momental, thres, kernel_size, loss_fn):
    batch_size = x_nature.shape[0]
    g = torch.zeros_like(x_nature)
    delta = torch.zeros_like(x_nature).cuda()
    eps_list = [(idx + 1) * (max_eps / intervals) for idx in range(intervals)]
    mask = torch.ones((batch_size, )).bool()
    for eps in eps_list:
        eps /= 255.0
        step_size = 1.25 * eps / num_steps
        delta = Variable(delta.data, requires_grad=True)

        for _ in range(num_steps):
            delta.requires_grad_()
            adv = x_nature + delta
            adv = torch.clamp(adv, 0, 1)
            with torch.enable_grad():
                #ensem_logits = model(adv, diversity=True)
                ensem_logits = model(adv)
                loss = loss_fn(ensem_logits, y)

            PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
            PGD_grad = F.conv2d(PGD_grad, weight=get_kernel(kernel_size), stride=(
                1, 1), groups=3, padding=(kernel_size - 1) // 2)
            PGD_noise = PGD_grad / \
                torch.abs(PGD_grad).mean(dim=(1, 2, 3), keepdim=True)
            g[mask] = g[mask] * momental + PGD_noise[mask]
            delta = Variable(delta.data + step_size *
                             torch.sign(g), requires_grad=True)
            delta = Variable(torch.clamp(
                delta.data, -eps, eps), requires_grad=True)

        g.zero_()
        with torch.no_grad():
            tmp = x_nature + delta
            tmp = torch.clamp(tmp, 0, 1)
            #output = eval_model(tmp, diversity=False).detach()
            output = eval_model(tmp).detach()
        prob = F.softmax(output, dim=1)
        conf = prob[np.arange(batch_size), y.long()]
        mask = (conf >= thres)

        # early stopping
        if mask.sum() == 0:
            break

    X_pgd = Variable(x_nature + delta, requires_grad=False)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
    #budget = torch.abs(X_pgd - x_nature).reshape(batch_size, -1).max(dim = -1)[0]
    return X_pgd#, budget

def ga_dmi_fgsm_wrapper(model: nn.Module,
                         inputs: Tensor,
                         labels: Tensor,
                         targets: Optional[Tensor] = None,
                         targeted: bool = False,
                         max_eps: float = 20,
                         intervals: int = 5,
                         num_steps: int = 10,
                         momental: float = 1.0,
                         thres: float = 0.2,
                         kernel_size: int = 5,
                         loss_fn: str = 'ce',
                         target_model: Optional[nn.Module] = None) -> Tensor:
    # loss_fn
    if loss_fn == "ce":
        loss = CrossEntropyLoss()
    elif loss_fn == "margin":
        loss = MarginLoss()
    else:
        raise Exception("invalid loss function!")
    #target model
    if target_model == None:
        target_model = model
    return GA_DMI_fgsm(model=model, eval_model=target_model, x_nature=inputs, y=labels, max_eps=max_eps, intervals=intervals,
                 num_steps=num_steps, momental=momental, thres=thres, kernel_size=kernel_size, loss_fn=loss)

class CrossEntropyLoss(nn.Module):
    """
    cross entropy loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')


class MarginLoss(nn.Module):
    """
    top-5 margin loss
    """

    def __init__(self, kappa=float('inf'), k = 5):
        super().__init__()
        self.kappa = kappa
        self.k = k

    def forward(self, logits, labels, conf=1):
        onehot_label = F.one_hot(labels, num_classes=1000).float()
        true_logit5 = torch.sum(logits * onehot_label, dim=-1, keepdims=True)
        wrong_logit5, _idx = torch.topk(logits * (1-onehot_label) - onehot_label * 1e7, k=self.k, dim = 1)
        target_loss5 = torch.sum(F.relu(true_logit5 - wrong_logit5 + conf), dim = 1)
        return target_loss5

def input_diversity(input_tensor, target_size, diversity_scale=0.1, prob=0.7):
    upper_bound = int(target_size * (diversity_scale + 1.0))
    lower_bound = int(target_size * (1.0 - diversity_scale))
    rnd = np.floor(np.random.uniform(lower_bound, upper_bound, size=())).astype(np.int32).item()
    x_resize = F.interpolate(input_tensor, size=rnd)
    h_rem = upper_bound - rnd
    w_rem = upper_bound - rnd
    pad_top = np.floor(np.random.uniform(0, h_rem, size=())).astype(np.int32).item()
    pad_bottom = h_rem - pad_top
    pad_left = np.floor(np.random.uniform(0, w_rem, size=())).astype(np.int32).item()
    pad_right = w_rem - pad_left
    padded = F.pad(x_resize, (int(pad_top), int(pad_bottom),int(pad_left), int(pad_right), 0, 0, 0, 0))
    if torch.rand(1) <= prob:
        return F.interpolate(padded, size=target_size)
    else:
        return F.interpolate(input_tensor, size=target_size)
