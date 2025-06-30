import json
import warnings
from pathlib import Path
from typing import Mapping, Tuple, Union, List

import numpy as np

from .utils import Scenario, get_model_key, get_model_key_imagenet


def read_distances(info_file: Union[Path, str],
                   distance_type: str = 'best',
                   already_adv_distance: float = 0,
                   worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    threat_model = config['attack']['threat_model']

    if dataset == "cifar10":
        model = get_model_key(config['model']['name'])
    elif dataset == "imagenet":
        model = get_model_key_imagenet(config['model']['name'])
    else:
        raise ValueError("dataset not support")
    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get hashes and distances for the adversarial examples
    hashes = info['hashes']
    distances = np.array(info['best_optim_distances' if distance_type == 'best' else 'distances'][threat_model])
    ori_success = np.array(info['ori_success'])
    adv_success = np.array(info['adv_success'])

    # check that adversarial examples have 0 distance for adversarial clean samples
    if (n := np.count_nonzero(distances[ori_success])):
        warnings.warn(f'{n} already adversarial clean samples have non zero perturbations in {info_file}.')

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    distances[~adv_success] = worst_case_distance
    distances[ori_success] = already_adv_distance

    # store results
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    hash_distances = {hash: distance for (hash, distance) in zip(hashes, distances)}
    return scenario, hash_distances



def read_distances_with_trans(info_file: Union[Path, str],
                   target_models: List[str],
                   distance_type: str = 'best',
                   already_adv_distance: float = 0,
                   worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float], dict]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    threat_model = config['attack']['threat_model']
    if dataset == "cifar10":
        model = get_model_key(config['model']['name'])
    elif dataset == "imagenet":
        model = get_model_key_imagenet(config['model']['name'])
    else:
        raise ValueError("dataset not support")


    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get hashes and distances for the adversarial examples
    hashes = info['hashes']
    all_distances = {}
    distances = np.array(info['best_optim_distances' if distance_type == 'best' else 'distances'][threat_model])
    ori_success = np.array(info['ori_success'])
    adv_success = np.array(info['adv_success'])

    # check that adversarial examples have 0 distance for adversarial clean samples
    if (n := np.count_nonzero(distances[ori_success])):
        warnings.warn(f'{n} already adversarial clean samples have non zero perturbations in {info_file}.')

    for target_model in target_models:
        if f"trans_to_{target_model}" not in info.keys():
            continue
        t_distances = np.array(info["trans_to_"+target_model]['best_optim_distances' if distance_type == 'best' else 'distances'][threat_model])
        t_hash = info["trans_to_"+target_model]['hashes']
        t_ori_success = np.array(info["trans_to_"+target_model]['ori_success'])
        t_adv_success = np.array(info["trans_to_"+target_model]['adv_success'])
        t_distances[~t_adv_success] = worst_case_distance
        t_distances[t_ori_success] = already_adv_distance
        h_d = {hash: distance for (hash, distance) in zip(t_hash, t_distances)}
        all_distances[target_model] = h_d.copy()
        del t_distances
        del h_d
        del t_hash

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    distances[~adv_success] = worst_case_distance
    distances[ori_success] = already_adv_distance

    # store results
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    hash_distances = {hash: distance for (hash, distance) in zip(hashes, distances)}

    #get time per example
    times = info["times"]
    per_time = sum(times)/len(hashes)

    return scenario, hash_distances, all_distances, per_time

def read_info(info_file: Union[Path, str],
                 already_adv_distance: float = 0,
                 worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    threat_model = config['attack']['threat_model']
    if dataset == "cifar10":
        model = get_model_key(config['model']['name'])
    elif dataset == "imagenet":
        model = get_model_key_imagenet(config['model']['name'])
    else:
        raise ValueError("dataset not support")

    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get distances for the adversarial examples wrt the given threat model
    for key in info.keys():
        if isinstance(info[key], list):
            info[key] = np.array(info[key])

    ori_success = info['ori_success']
    adv_success = info['adv_success']


    # replace distances with inf for failed attacks and 0 for already adv clean samples
    for distance_type in ['distances', 'best_optim_distances']:
        info[distance_type] = np.array(info[distance_type][threat_model])
        info[distance_type][~adv_success] = worst_case_distance
        info[distance_type][ori_success] = already_adv_distance
    info['adv_valid_success'] = adv_success & (~ori_success)

    # store results
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    return scenario, info

def read_info_with_trans(info_file: Union[Path, str],
                 target_models: List[str],
                 already_adv_distance: float = 0,
                 worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    threat_model = config['attack']['threat_model']

    if dataset == "cifar10":
        model = get_model_key(config['model']['name'])
    elif dataset == "imagenet":
        model = get_model_key_imagenet(config['model']['name'])
    else:
        raise ValueError("dataset not support")

    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get distances for the adversarial examples wrt the given threat model
    for key in info.keys():
        if isinstance(info[key], list):
            info[key] = np.array(info[key])

    ori_success = info['ori_success']
    adv_success = info['adv_success']


    # replace distances with inf for failed attacks and 0 for already adv clean samples
    for distance_type in ['distances', 'best_optim_distances']:
        info[distance_type] = np.array(info[distance_type][threat_model])
        info[distance_type][ori_success] = already_adv_distance
        info[distance_type][~adv_success] = worst_case_distance
    info['adv_valid_success'] = adv_success & (~ori_success)

    # store results
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    return scenario, info
