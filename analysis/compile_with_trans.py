import argparse
import json
import os
import pathlib
from typing import List


from .read import read_distances_with_trans
from .utils import Scenario


def compile_scenario_with_trans(path: pathlib.Path, scenario: Scenario, distance_type: str,
                                target_models: List[str], recompile_all: bool = False) -> None:
    # find completed experiment results
    scenario_path = path / scenario.dataset / scenario.threat_model / scenario.model.lower()
    info_files = list(scenario_path.glob(os.path.join('**', 'info.json')))
    if len(info_files) == 0:
        return

    # open previous file if already existing
    best_distances_path = path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}-with_trans.json'
    best_distances = {}
    best_distances_target = {}  # dict.fromkeys(target_models)
    for target_model in target_models:
        best_distances_target[target_model] = {}
    if best_distances_path.exists() and not recompile_all:
        with open(best_distances_path, 'r') as f:
            best_distances = json.load(f)

    # compile best distances
    for info_file in info_files:
        _, hash_distances, target_distances, _ = read_distances_with_trans(info_file=info_file,
                                                                        target_models=target_models,
                                                                        distance_type=distance_type,
                                                                        worst_case_distance=100000)
        for (hash, distance) in hash_distances.items():
            best_distance = best_distances.get(hash, float('inf'))
            if distance < best_distance:
                best_distances[hash] = distance
        for target_model in target_models:
            if target_model not in target_distances:
                continue    #0623
            for (hash, distance) in target_distances[target_model].items():
                bd = best_distances_target[target_model].get(hash, float('inf'))
                if distance < bd:
                    best_distances_target[target_model][hash] = distance

    best_distances_target["white_box"] = best_distances

    with open(best_distances_path, 'w') as f:
        json.dump(best_distances_target, f, indent=4)


def get_one_attack_result_with_trans(path: pathlib.Path, scenario: Scenario, distance_type: str,
                                target_models: List[str], attack_type: str, recompile_all: bool = False):
    # find completed experiment results
    batch_size = f'batch_size_{scenario.batch_size}'
    scenario_path = path / scenario.dataset / scenario.threat_model / scenario.model.lower() / batch_size / attack_type
    info_files = list(scenario_path.glob(os.path.join('**', 'info.json')))
    if len(info_files) == 0:
        return

    # open previous file if already existing
    best_distances = {}
    best_distances_target = {}  # dict.fromkeys(target_models)
    for target_model in target_models:
        best_distances_target[target_model] = {}

    # compile best distances
    for info_file in info_files:
        _, hash_distances, target_distances, per_time = read_distances_with_trans(info_file=info_file,
                                                                        target_models=target_models,
                                                                        distance_type=distance_type,
                                                                        worst_case_distance=100000)
        for (hash, distance) in hash_distances.items():
            best_distance = best_distances.get(hash, float('inf'))
            if distance < best_distance:
                best_distances[hash] = distance
        for target_model in target_models:
            if target_model not in target_distances:
                continue
            for (hash, distance) in target_distances[target_model].items():
                bd = best_distances_target[target_model].get(hash, float('inf'))
                if distance < bd:
                    best_distances_target[target_model][hash] = distance

    best_distances_target["white_box"] = best_distances
    return best_distances_target, per_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compile results from several attacks')

    parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
    parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
    parser.add_argument('--recompile-all', '--ra', action='store_true',
                        help='Ignores previous best distance file and recompile it from scratch.')

    args = parser.parse_args()

    # check that result directory exists
    result_path = pathlib.Path(args.dir)
    assert result_path.exists()  # find info files corresponding to finished experiments

    dataset = args.dataset or '*'
    threat_model = args.threat_model or '*'
    model = args.model or '*'

    scenarios = result_path.glob(os.sep.join((dataset, threat_model, model)))
    # for scenario_dir in scenarios:
    #     scenario_dataset, scenario_threat_model, scenario_model = scenario_dir.parts[-3:]
    #     scenario = Scenario(dataset=scenario_dataset, threat_model=scenario_threat_model, model=scenario_model)
    #     compile_scenario_with_trans(path=result_path, scenario=scenario, recompile_all=args.recompile_all)
