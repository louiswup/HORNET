from functools import partial
from typing import Callable, Optional

from .auto_pgd import apgd_attack, apgd_minimal_wrapper, apgd_t_attack
from .deepfool import deepfool_attack
from .fast_adaptive_boundary import fab_attack
from .fast_minimum_norm import fmn_attack
from .pgd_lzero import PGD0_minimal
from .sigma_zero import sigma_zero
from .trust_region import tr_attack
from .. import minimal_init_eps, minimal_search_steps, max_bound
from .ga_fgsm import ga_dmi_fgsm_wrapper, ga_tdmi_fgsm_wrapper

_prefix = 'original'


def _wrapper(attack, **kwargs): return attack(**kwargs)


def original_apgd():
    name = 'apgd'
    source = 'original'
    threat_model = 'linf'  # available: 'l1', 'l2', 'linf'
    num_steps = 100
    num_restarts = 1
    epsilon = 0.3
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


def original_apgd_l1():
    name = 'apgd'
    source = 'original'
    threat_model = 'l1'
    num_steps = 100
    num_restarts = 1
    epsilon = 10
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = True  # set True with L1 norm


def get_original_apgd(threat_model: str, num_steps: int, num_restarts: int, epsilon: float, loss: str, rho: float,
                      use_largereps: bool) -> Callable:
    return partial(apgd_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts, eps=epsilon,
                   loss=loss, rho=rho, use_largereps=use_largereps)


def original_apgd_hornet():
    name = 'apgd_hornet'
    source = 'original'
    threat_model = 'linf'  # available: 'l1', 'l2', 'linf'
    num_steps = 20#100
    num_restarts = 1
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


def original_apgd_l1_hornet():
    name = 'apgd_hornet'
    source = 'original'
    threat_model = 'l1'
    num_steps = 20#100
    num_restarts = 1
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = True  # set True with L1 norm

def original_apgd_l2_hornet():
    name = 'apgd_hornet'
    source = 'original'
    threat_model = 'l2'
    num_steps = 20#100
    num_restarts = 1
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


def get_original_apgd_hornet(threat_model: str, num_steps: int, num_restarts: int, loss: str, rho: float,
                              use_largereps: bool,
                              init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    attack = partial(apgd_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts, loss=loss,
                     rho=rho, use_largereps=use_largereps)
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = max_bound[threat_model]
    return partial(apgd_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def original_apgd_t():
    name = 'apgd_t'
    source = 'original'
    threat_model = 'linf'  # available: 'l1', 'l2', 'linf'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    epsilon = 0.3
    rho = .75
    use_largereps = False  # set True with L1 norm


def original_apgd_t_l1():
    name = 'apgd_t'
    source = 'original'
    threat_model = 'l1'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    epsilon = 10
    rho = .75
    use_largereps = True  # set True with L1 norm


def get_original_apgd_t(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int, epsilon: float,
                        rho: float, use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                   n_target_classes=num_target_classes, eps=epsilon, rho=rho, use_largereps=use_largereps)


def original_apgd_t_hornet():
    name = 'apgd_t_hornet'
    source = 'original'
    threat_model = 'linf'  # available: 'l1', 'l2', 'linf'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    rho = .75
    use_largereps = False  # set True with L1 norm

def original_apgd_t_l1_hornet():
    name = 'apgd_t_hornet'
    source = 'original'
    threat_model = 'l1'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    rho = .75
    use_largereps = True  # set True with L1 norm


def get_original_apgd_t_hornet(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int,
                                rho: float, use_largereps: bool,
                                init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    attack = partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                     n_target_classes=num_target_classes, rho=rho, use_largereps=use_largereps)
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = max_bound[threat_model]
    return partial(apgd_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def original_deepfool():
    name = 'deepfool'
    source = 'original'
    threat_model = 'l2'
    num_classes = 10  # number of classes to test gradient (can be different from the number of classes of the model)
    overshoot = 0.02
    num_steps = 50


def get_original_deepfool(num_classes: int, overshoot: float, num_steps: int) -> Callable:
    return partial(deepfool_attack, num_classes=num_classes, overshoot=overshoot, max_iter=num_steps)


def original_fab():
    name = 'fab'
    source = 'original'
    threat_model = 'linf'  # available: 'l1', 'l2', 'linf'
    num_restarts = 1
    num_steps = 100
    epsilon = None
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted_variant = False
    n_target_classes = 9


def get_original_fab(threat_model: str, num_restarts: int, num_steps: int, epsilon: Optional[float], alpha_max: float,
                     eta: float, beta: float, targeted_variant: bool, n_target_classes: int) -> Callable:
    return partial(fab_attack, threat_model=threat_model, n_restarts=num_restarts, n_iter=num_steps, eps=epsilon,
                   alpha_max=alpha_max, eta=eta, beta=beta, targeted_variant=targeted_variant,
                   n_target_classes=n_target_classes)


def original_fmn():
    name = 'fmn'
    source = 'original'
    threat_model = 'l2'  # available: 'l0', 'l1', 'l2', 'linf'
    num_steps = 1000
    max_step_size = 1
    gamma = 0.05


def original_fmn_linf():
    name = 'fmn'
    source = 'original'
    threat_model = 'linf'
    num_steps = 1000
    max_step_size = 10
    gamma = 0.05


def get_original_fmn(threat_model: str, num_steps: int, max_step_size: float, gamma: float) -> Callable:
    return partial(fmn_attack, threat_model=threat_model, steps=num_steps, max_stepsize=max_step_size, gamma=gamma)


def original_tr():
    name = 'tr'
    source = 'original'
    threat_model = 'linf'  # available: 'l2', 'linf'
    adaptive = False
    epsilon = 0.001
    c = 9
    num_steps = 100


def original_tr_adaptive():
    name = 'tr'
    source = 'original'
    threat_model = 'linf'  # available: 'l2', 'linf'
    adaptive = True
    epsilon = 0.001
    c = 9
    num_steps = 100


def get_original_tr(threat_model: str, adaptive: bool, epsilon: float, c: int, num_steps: int) -> Callable:
    return partial(tr_attack, threat_model=threat_model, adaptive=adaptive, eps=epsilon, c=c, iter=num_steps)


def original_sigma_zero():
    name = 'sigma_zero'
    source = 'original'
    threat_model = 'l0'  # available: 'l0', 'l1', 'l2', 'linf'
    num_steps = 100
    lr = 1.0
    sigma = 1e-3
    thr_0 = 0.3
    thr_lr = 0.01
    binary_search_steps = 10


def get_original_sigma_zero(threat_model: str, num_steps: int, lr: float, sigma: float, thr_0: float, thr_lr: float,
                            binary_search_steps: int) -> Callable:
    return partial(sigma_zero, steps=num_steps, lr=lr, sigma=sigma, thr_0=thr_0, thr_lr=thr_lr,
                   binary_search_steps=binary_search_steps)


def original_pgd0_hornet():
    name = 'pgd0_hornet'
    source = 'original'
    threat_model = 'l0'
    n_restarts = 1
    num_steps = 100
    step_size = 120000 / 255
    kappa = -1
    epsilon = -1
    init_eps = 512
    search_steps = 9


def get_original_pgd0_hornet(threat_model: str, num_steps, step_size, kappa, epsilon, n_restarts,
                              init_eps: Optional[int] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    return partial(PGD0_minimal, search_steps=search_steps, num_steps=num_steps, step_size=step_size, kappa=kappa,
                   epsilon=epsilon, init_eps=init_eps, n_restarts=n_restarts)

def original_ga_tdmi_fgsm():
    name = 'ga_tdmi_fgsm'
    source = 'original'
    threat_model = 'linf'
    max_eps = 255
    intervals = 5
    num_steps = 10
    momental = 1.0
    thres = 0.2
    kernel_size = 5
    loss_fn = 'ce'

def get_original_ga_tdmi_fgsm(threat_model: str, max_eps, intervals,
                         num_steps, momental, thres, kernel_size, loss_fn) -> Callable:
    return partial(ga_tdmi_fgsm_wrapper, max_eps=max_eps, intervals=intervals,
                   num_steps=num_steps, momental=momental, thres=thres, kernel_size=kernel_size, loss_fn=loss_fn)

def original_ga_dmi_fgsm():
    name = 'ga_dmi_fgsm'
    source = 'original'
    threat_model = 'linf'
    max_eps = 255
    intervals = 5
    num_steps = 10
    momental = 1.0
    thres = 0.2
    kernel_size = 5
    loss_fn = 'ce'

def get_original_ga_dmi_fgsm(threat_model: str, max_eps, intervals,
                         num_steps, momental, thres, kernel_size, loss_fn) -> Callable:
    return partial(ga_dmi_fgsm_wrapper, max_eps=max_eps, intervals=intervals,
                   num_steps=num_steps, momental=momental, thres=thres, kernel_size=kernel_size, loss_fn=loss_fn)
