from collections import OrderedDict,defaultdict
from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, Optional, Union

import torch
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from sacred import Experiment
from sacred.observers import FileStorageObserver

from .attacks.ingredient import attack_ingredient, get_attack
from .datasets.ingredient import dataset_ingredient, get_loader
from .models.ingredient import get_model, model_ingredient, get_local_model, get_robustbench_model, get_original_model
from .utils import run_attack, set_seed

ex = Experiment('attack_evaluation', ingredients=[dataset_ingredient, model_ingredient, attack_ingredient])


@ex.config
def config():
    cpu = False  # force experiment to run on CPU
    save_adv = False  # save the inputs and perturbed inputs; not to be used with large datasets
    cudnn_flag = 'deterministic'# choose between "deterministic" and "benchmark"
    targets = None#target attack


@ex.named_config
def save_adv():  # act as a flag
    save_adv = True


@ex.option_hook
def modify_filestorage(options):
    if (file_storage := options['--file_storage']) is None:
        return

    update = options['UPDATE']

    # find dataset, model and attack names from CLI
    names = {}
    for ingredient in (model_ingredient, attack_ingredient):
        ingredient_name = ingredient.path
        prefix = ingredient_name + '.'
        ingredient_updates = list(filter(lambda s: s.startswith(prefix) and '=' not in s, update))
        if (n := len(ingredient_updates)) != 1:
            raise ValueError(f'Incorrect {ingredient_name} configuration: {n} (!=1) named configs specified.')
        named_config = ingredient_updates[0].removeprefix(prefix)
        # names.append(ingredient.named_configs[named_config]()['name'])
        names[ingredient_name] = named_config

    # get dataset from model named config
    dataset = model_ingredient.named_configs[names['model']]()['dataset']

    # find threat model
    attack_updates = list(filter(lambda s: s.startswith('attack.') and 'threat_model=' in s, update))
    if len(attack_updates):
        threat_model = attack_updates[-1].split('=')[-1]
    else:
        threat_model = ingredient.named_configs[named_config]()['threat_model']

    batch_size_update = list(filter(lambda s: "batch_size" in s, update))
    if len(batch_size_update):
        batch_size = batch_size_update[-1].split('=')[-1]
    else:
        batch_size = dataset_ingredient.configurations[0]()['batch_size']
    batch_name = f'batch_size_{batch_size}'

    # insert threat model and batch_size at desired position for folder structure
    subdirs = [dataset, threat_model, names['model'], batch_name, names['attack']]
    options['--file_storage'] = Path(file_storage).joinpath(*subdirs).as_posix()


metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])
_model_getters = {
    'local': get_local_model,
    'robustbench': get_robustbench_model,
    'original': get_original_model
}



@ex.automain
def main(cpu: bool,
         cudnn_flag: str,
         save_adv: bool,
         targets: int,
         _config, _run, _log, _seed):
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    setattr(torch.backends.cudnn, cudnn_flag, True)

    set_seed(_seed)
    print(f'Running experiments with seed {_seed}')

    threat_model = _config['attack']['threat_model']
    loader = get_loader(dataset=_config['model']['dataset'])
    attack = get_attack()
    model = get_model()
    model.to(device)


    if len(loader) == 0:  # end experiment if there are no inputs to attack
        return

    # find the current folder where the artifacts are saved
    file_observers = [obs for obs in _run.observers if isinstance(obs, FileStorageObserver)]
    save_dir = file_observers[0].dir if len(file_observers) else None

    attack_data = run_attack(model=model, loader=loader, attack=attack, metrics=metrics, threat_model=threat_model,
                             return_adv=True, debug=_run.debug)
    recur_dict = lambda: defaultdict(recur_dict)
    trans_data = recur_dict()
    target_models_list = _config["model"]["target_models"]

    for target_model in target_models_list:
        t_model_info = model_ingredient.named_configs[target_model]()
        t_model = _model_getters[t_model_info['source']](name=t_model_info['name'], dataset=t_model_info['dataset'], threat_model=t_model_info['threat_model'])
        t_model.eval()
        t_model.to(device)
        accuracies, ori_success, adv_success = [],[],[]
        attack_data_t = run_attack(model=model, loader=loader, attack=attack, metrics=metrics, threat_model=threat_model,
                                             return_adv=True, target_model=t_model, debug=_run.debug)
        for i,inputs in enumerate(attack_data_t['adv_inputs']):
            labels = attack_data_t['labels'][i].to(device)
            logits = t_model(inputs.to(device))
            ori_inputs = attack_data_t['inputs'][i].to(device)
            ori_logits = t_model(ori_inputs.to(device))

            predictions = ori_logits.argmax(dim=1)
            accuracies.extend((predictions == labels).cpu().tolist())
            success = (predictions == targets) if targets is not None else (predictions != labels)
            ori_success.extend(success.cpu().tolist())

            adv_predictions = logits.argmax(dim=1)
            success = (adv_predictions == targets) if targets is not None else (adv_predictions != labels)
            adv_success.extend(success.cpu().tolist())
        
        trans_data["trans_to_"+target_model]["accuracy"] = (sum(accuracies) / len(accuracies))
        trans_data["trans_to_"+target_model]["ori_success"] = ori_success
        trans_data["trans_to_"+target_model]["adv_success"] = adv_success
        trans_data["trans_to_"+target_model]["distances"] = attack_data_t["distances"]
        trans_data["trans_to_"+target_model]["best_optim_distances"] = attack_data_t["distances"]
        trans_data["trans_to_"+target_model]["hashes"] = attack_data_t["hashes"]
        trans_data["trans_to_"+target_model]["ASR"] = sum(adv_success) / len(adv_success)

        
    attack_data.update(trans_data)
    if save_adv and save_dir is not None:
        torch.save(attack_data, Path(save_dir) / f'attack_data.pt')

    if 'inputs' in attack_data.keys():
        del attack_data['inputs'], attack_data['adv_inputs']
    _run.info = attack_data

    if _run.debug:
        pprint(_run.info)

