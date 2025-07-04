import argparse
import json
import os
import pathlib

from .read import read_distances
from .utils import Scenario


def compile_scenario(path: pathlib.Path, scenario: Scenario, distance_type: str, recompile_all: bool = False) -> None:
    # find completed experiment results
    scenario_path = path / scenario.dataset / scenario.threat_model / scenario.model.lower()
    info_files = list(scenario_path.glob(os.path.join('**', 'info.json')))
    if len(info_files) == 0:
        return

    # open previous file if already existing
    best_distances_path = path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}.json'
    best_distances = {}
    if best_distances_path.exists() and not recompile_all:
        with open(best_distances_path, 'r') as f:
            best_distances = json.load(f)

    # compile best distances
    for info_file in info_files:
        hash_distances = read_distances(info_file=info_file, distance_type=distance_type)[1]
        for hash, distance in hash_distances.items():
            best_distance = best_distances.get(hash, float('inf'))
            if distance < best_distance:
                best_distances[hash] = distance


    with open(best_distances_path, 'w') as f:
        json.dump(best_distances, f, indent=4)


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
    for scenario_dir in scenarios:
        scenario_dataset, scenario_threat_model, scenario_model = scenario_dir.parts[-3:]
        scenario = Scenario(dataset=scenario_dataset, threat_model=scenario_threat_model, model=scenario_model)
        compile_scenario(path=result_path, scenario=scenario, recompile_all=args.recompile_all)
