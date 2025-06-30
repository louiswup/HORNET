import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict
from .utils import Scenario

import numpy as np
from matplotlib import pyplot as plt, ticker

from .compile_with_trans import compile_scenario_with_trans, get_one_attack_result_with_trans
from .read import read_distances, read_info
from .utils import top_k_attacks

from tabulate import tabulate

threat_model_labels = {
    'l0': r'$\ell_0$',
    'l1': r'$\ell_1$',
    'l2': r'$\ell_2$',
    'linf': r'$\ell_{\infty}$',
}

max_bound = {
    'l0': 3072,
    'l1': 3072,
    'l2': 55.426,
    'linf': 1.0,
}

ROUND = lambda x: np.around(x, 3)
TOLERANCE = 1e-04
# smooth_step = 0.02#for fixed-norm attack
# smooth_step = 0#for mini-norm attack
#source_models = ["standard", "zhang_2020_small", "stutz_2020", "xiao_2020", "wang_2023_small"]
#target_models = ["cohen_2019_certified", "yang_2019_MENet"]

#save subplot
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path+fig_name, bbox_inches=extent,dpi=800)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot results')

    parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
    parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
    parser.add_argument('--library', '-l', type=str, default=None, help='Library for which to plot results')
    parser.add_argument('--K', '-k', type=int, default=None, help='Top K attacks to show')
    parser.add_argument('--batch_size', '-bs', type=int, default=None, help="Batch size for which to plot the results")
    parser.add_argument('--info-files', '--if', type=str, nargs='+', default=None,
                        help='List of info files to plot from.')
    parser.add_argument('--distance_type', '-dist', type=str, default='best', choices=['best', 'actual'],
                        help='Define distances to plot results')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='Suffix for the name of the plot')
    parser.add_argument('--smooth_step', '-ss', type=float, default=0,
                        help='0 for minimal attack, 0.02 for fixed-norm attack')
    parser.add_argument('--source_models', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--target_models', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()

    source_models = args.source_models
    target_models = args.target_models
    # check that result directory exists
    result_path = pathlib.Path(args.dir)
    assert result_path.exists()

    # check source model exists
    threat_model = args.threat_model or '*'
    threat_model_lst = threat_model_labels.keys() if threat_model == '*' else [threat_model]
    Table = {}
    ensemble_result = {}
    ensemble_attack = []
    ensemble_times = {}

    for key in threat_model_lst:
        # Table[key] = [
        #    ['Dataset', 'BatchSize', 'Threat', 'Source Model', 'Attack', 'ASR', 'Optimality'] + target_models + ['Average transfer optimality']]
        kk = []
        for model in target_models:
            kk.append(model + " TSR(FR)")
            kk.append(model + " S_APR")
            kk.append(model + " S_total")
            kk.append(model + " median_norm")
            kk.append(model + " Optimality")
        Table[key] = [
            ['Dataset', 'BatchSize', 'Threat', 'Source Model', 'Attack'] + kk + ['per-time','Average transfer optimality']]

    for source_model in source_models:
        '''target_models = ["standard", "zhang_2020_small", "stutz_2020", "xiao_2020", "wang_2023_small"]
        if source_model in target_models:
            target_models.remove(source_model)
        else:
            print("error: source model does not exist")
            assert 1'''

        distance_type = args.distance_type
        smooth_step = args.smooth_step

        to_plot = defaultdict(list)
        if args.info_files is not None:
            info_files_paths = args.info_files
            info_files = [pathlib.Path(info_file) for info_file in info_files_paths]
            assert all(info_file.exists() for info_file in info_files)
        else:
            dataset = args.dataset or '*'
            model = source_model or '*'
            # library = f'{args.library}_*/**' if args.library else '**'
            library = f'{args.library}/**' if args.library else '**'
            batch_size = f'batch_size_{args.batch_size}' if args.batch_size else '*'
            info_files_paths = os.sep.join((dataset, threat_model, model, batch_size, library, 'info.json'))
            info_files = result_path.glob(info_files_paths)

        for info_file in info_files:
            scenario, info = read_info(info_file)
            to_plot[scenario].append((info_file.parent, info))
            scenario_ensemble = Scenario(dataset=scenario.dataset, batch_size=scenario.batch_size,
                                         threat_model=scenario.threat_model, model="ensemble")
            if scenario_ensemble not in ensemble_result:
                ensemble_times[scenario_ensemble] = {}
                ensemble_result[scenario_ensemble] = {}
                for model in target_models:
                    ensemble_result[scenario_ensemble][model] = {}
                    ensemble_result[scenario_ensemble][model]["best"] = {}
                    ensemble_times[scenario_ensemble][model] = {}

        for scenario in to_plot.keys():
            best_distances_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}-with_trans.json'
            if not best_distances_file.exists():
                # print("Compiling ", scenario)
                warnings.warn(f'Best distances files {best_distances_file} does not exist for scenario {scenario}.')
                warnings.warn(f'Compiling best distances file for scenario {scenario}')
                compile_scenario_with_trans(path=result_path, target_models=target_models,
                                            scenario=scenario, distance_type=distance_type)

            with open(best_distances_file, 'r') as f:
                data = json.load(f)
            best_distances = list(data["white_box"].values())
            clip_num = (np.array(best_distances) > max_bound[scenario.threat_model]).sum()
            distances, counts = np.unique(np.array(best_distances).clip(min=None, max=max_bound[scenario.threat_model]), return_counts=True)
            counts[-1] = counts[-1] - clip_num
            robust_acc = 1 - counts.cumsum() / len(best_distances)

            # get quantities for optimality calculation
            clean_acc = np.count_nonzero(best_distances) / len(best_distances)
            max_dist = np.amax(distances)
            best_area = np.trapz(robust_acc, distances)

            scenario_ensemble = Scenario(dataset=scenario.dataset, batch_size=scenario.batch_size,
                                         threat_model=scenario.threat_model, model="ensemble")
            t_clean_acc, t_max_dist, t_best_area = [], [], []
            for target_model in target_models:
                for key, value in data[target_model].items():
                    best_value = ensemble_result[scenario_ensemble][target_model]['best'].get(key, float('inf'))
                    if value < best_value:
                        ensemble_result[scenario_ensemble][target_model]['best'][key] = value

                t_best_distances = list(data[target_model].values())
                clip_num = (np.array(t_best_distances) > max_bound[scenario.threat_model]).sum()
                t_distances, t_counts = np.unique(np.array(t_best_distances).clip(min=None, max=max_bound[scenario.threat_model]), return_counts=True)
                t_counts[-1] = t_counts[-1] - clip_num
                t_robust_acc = 1 - t_counts.cumsum() / len(t_best_distances)  # modify
                t_clean_acc.append(np.count_nonzero(t_best_distances) / len(t_best_distances))
                t_max_dist.append(np.amax(t_distances))
                t_best_area.append(np.trapz(t_robust_acc, t_distances))
         
            bd_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-E.json'
            '''with open(bd_file, 'r') as f:
                bd_data = json.load(f)
            i=0
            for target_model in target_models:
                if target_model != source_model:
                    t_clean_acc.append(bd_data['t_clean_acc'][i])
                    t_max_dist.append(bd_data['t_max_dist'][i])
                    t_best_area.append(bd_data['t_best_area'][i])
                    i+=1'''

            attacks_to_plot = {}
            for attack_folder, info in sorted(to_plot[scenario]):
                attack_label = attack_folder.relative_to(attack_folder.parents[1]).as_posix()
                attack_label = attack_label.split('/')[0]
                if attack_label in attacks_to_plot:
                    continue
                one_attack_result,per_time = get_one_attack_result_with_trans(path=result_path, target_models=target_models,
                                                                     attack_type=attack_label,
                                                                     scenario=scenario, distance_type=distance_type)
                if attack_label not in ensemble_attack:
                    ensemble_attack.append(attack_label)
                    for model in target_models:
                        ensemble_result[scenario_ensemble][model][attack_label] = {}
                        ensemble_times[scenario_ensemble][model][attack_label] = per_time
                row = list(scenario) + [attack_label]
                average_trans = 0.0
                i = 0
                for target_model in target_models:
                    if target_model == source_model:
                        adv_distances = np.array(list(one_attack_result['white_box'].values()))
                        suc_fool = (adv_distances < max_bound[scenario.threat_model]) & (adv_distances > 0)
                        ASR = 100.0 * suc_fool.sum() / (adv_distances > 0).sum()
                        if scenario.threat_model=="linf": 
                            eps_fool = 255.0 * adv_distances[suc_fool]
                        elif scenario.threat_model=="l2":
                            eps_fool = 16 * adv_distances[suc_fool]
                        else:
                            eps_fool = adv_distances[suc_fool]
                        aps_fool = 1.0 / eps_fool
                        S_APR = aps_fool.mean()
                        S_total = ASR * S_APR
                        median = np.median(adv_distances)
                        # optimality
                        clip_num = (adv_distances > max_dist).sum()
                        distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist),
                                                              return_counts=True)
                        #counts[-1] = counts[-1] - int(clip_num*(max_dist/max_bound[scenario.threat_model]))
                        counts[-1] = counts[-1] - clip_num
                        robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

                        area = np.trapz(robust_acc_clipped, distances_clipped)
                        optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)
                        if optimality<0:
                            optimality=0
                        row = row + [np.around(ASR, 1), np.around(S_APR, 4), np.around(S_total, 2), ROUND(median), ROUND(optimality)]
                    else:
                        for key, value in one_attack_result[target_model].items():
                            best_value = ensemble_result[scenario_ensemble][target_model][attack_label].get(key, float(
                                'inf'))
                            if value < best_value:
                                ensemble_result[scenario_ensemble][target_model][attack_label][key] = value
                        adv_distances = np.array(list(one_attack_result[target_model].values()))
                        suc_fool = (adv_distances < max_bound[scenario.threat_model]) & (adv_distances > 0)
                        ASR = 100.0 * suc_fool.sum() / (adv_distances > 0).sum()
                        if scenario.threat_model=="linf": 
                            eps_fool = 255.0 * adv_distances[suc_fool]
                        elif scenario.threat_model=="l2":
                            eps_fool = 16 * adv_distances[suc_fool]
                        else:
                            eps_fool = adv_distances[suc_fool]
                        aps_fool = 1.0 / eps_fool
                        S_APR = aps_fool.mean()
                        S_total = ASR * S_APR
                        median = np.median(adv_distances)
                        # optimality
                        clip_num = (adv_distances > t_max_dist[i]).sum()
                        distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=t_max_dist[i]),
                                                              return_counts=True)
                        #counts[-1] = counts[-1] - int(clip_num*(t_max_dist[i]/max_bound[scenario.threat_model]))
                        counts[-1] = counts[-1] - clip_num
                        robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

                        area = np.trapz(robust_acc_clipped, distances_clipped)
                        optimality = 1 - (area - t_best_area[i]) / (t_clean_acc[i] * t_max_dist[i] - t_best_area[i])
                        if optimality<0:
                            optimality=0
                        row = row + [np.around(ASR, 1), np.around(S_APR, 4), np.around(S_total, 2), ROUND(median), ROUND(optimality)]
                        average_trans += optimality
                        i += 1

                row.append(ROUND(per_time))
                row.append(ROUND(average_trans / 2))
                attacks_to_plot[attack_label] = {'row': row, 'optimality': -average_trans, 'area': -average_trans}

            for attack_label in top_k_attacks(attacks_to_plot, k=args.K):
                atk = attacks_to_plot[attack_label]
                Table[scenario.threat_model].append(atk['row'])

            # Table[scenario.threat_model].append(row)
            # get quantities for optimality calculation

    for scenario, result in ensemble_result.items():
        attacks_to_plot = {}
        t_clean_acc, t_max_dist, t_best_area = [], [], []

        #plot fig
        fig, ax = plt.subplots(nrows=1, ncols=len(target_models),figsize=(14,4), layout='constrained')
        #plt.title(' - '.join(scenario), pad=10)

        i_ax = 0
        for model in target_models:
            t_best_distances = np.array(list(result[model]["best"].values()))
            clip_num = (t_best_distances > max_bound[scenario.threat_model]).sum()
            t_distances, t_counts = np.unique(t_best_distances.clip(min=None, max=max_bound[scenario.threat_model]),
                                              return_counts=True)
            t_counts[-1] = t_counts[-1] - clip_num
            t_robust_acc = 1 - t_counts.cumsum() / len(t_best_distances)  # modify
            t_clean_acc.append(1-t_counts[0] / len(t_best_distances))
            t_max_dist.append(np.amax(t_distances))
            t_best_area.append(np.trapz(t_robust_acc, t_distances))

            plot_xlim = np.amax(t_distances)
            ax[i_ax].plot(t_distances, t_robust_acc, linestyle='-', label=f'Best distances', c='k', linewidth=1)
            ax[i_ax].grid(True, linestyle='--', c='lightgray', which='major')
            ax[i_ax].yaxis.set_major_formatter(ticker.PercentFormatter(1))
            ax[i_ax].set_xlim(left=0, right=plot_xlim)
            ax[i_ax].set_ylim(bottom=0, top=1)
            ax[i_ax].spines['top'].set_visible(False)
            ax[i_ax].spines['right'].set_visible(False)

            ax[i_ax].set_ylabel('Robust Accuracy')
            ax[i_ax].set_xlabel(f'Perturbation Size {threat_model_labels[scenario.threat_model]}')

            ax[i_ax].annotate(text=f'Clean accuracy: {t_clean_acc[i_ax]:.2%}', xy=(0, t_clean_acc[i_ax]),
                        xytext=(ax[i_ax].get_xlim()[1] / 2, t_clean_acc[i_ax]), ha='left', va='center',
                        arrowprops={'arrowstyle': '-', 'linestyle': '--'})

            #ax[i_ax].set_title(f'Target model: {model}')
            i_ax += 1

        attacks = list(result[target_models[0]].keys())
        '''bd_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-E.json'
        bd_data = {'t_clean_acc':t_clean_acc,
                't_max_dist':t_max_dist,
                't_best_area':t_best_area}
        with open(bd_file, 'w') as f:
            json.dump(bd_data, f, indent=4)'''


        attacks.remove("best")
        for attack_label in attacks:
            row = list(scenario) + [attack_label]
            average_trans = 0.0
            i = 0
            time_sum = 0
            for model in target_models:
                adv_distances = np.array(list(result[model][attack_label].values()))
                suc_fool = (adv_distances < max_bound[scenario.threat_model]) & (adv_distances > 0)
                ASR = 100.0 * suc_fool.sum() / (adv_distances > 0).sum()
                if scenario.threat_model=="linf": 
                    eps_fool = 255.0 * adv_distances[suc_fool]
                elif scenario.threat_model=="l2":
                    eps_fool = 16 * adv_distances[suc_fool]
                else:
                    eps_fool = adv_distances[suc_fool]
                aps_fool = 1.0 / eps_fool
                S_APR = aps_fool.mean()
                S_total = ASR * S_APR
                median = np.median(adv_distances)
                # optimality
                clip_num = (adv_distances > t_max_dist[i]).sum()
                distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=t_max_dist[i]),
                                                      return_counts=True)
                #counts[-1] = counts[-1] - int(clip_num*(t_max_dist[i]/max_bound[scenario.threat_model]))
                counts[-1] = counts[-1] - clip_num
                robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

                area = np.trapz(robust_acc_clipped, distances_clipped)
                optimality = 1 - (area - t_best_area[i]) / (t_clean_acc[i] * t_max_dist[i] - t_best_area[i])
                if optimality<0:
                    optimality=0
                row = row + [np.around(ASR, 1), np.around(S_APR, 4), np.around(S_total, 2), ROUND(median), ROUND(optimality)]
                average_trans += optimality
                time_sum += ensemble_times[scenario][model][attack_label]
                i += 1

            row.append(ROUND(time_sum))
            row.append(ROUND(average_trans / 2))
            attacks_to_plot[attack_label] = {'row': row, 'optimality': -average_trans, 'area': -average_trans}

        for attack_label in top_k_attacks(attacks_to_plot, k=args.K):
            atk = attacks_to_plot[attack_label]
            Table[scenario.threat_model].append(atk['row'])

        #plot fig
        for attack_label in top_k_attacks(attacks_to_plot, k=5):
            i_ax = 0
            attack_label1 = attack_label.replace("_minimal","")
            attack_label1 = attack_label1.replace("_l1","")
            attack_label1 = attack_label1.replace("_20","")
            attack_label1 = "HNT-"+attack_label1.replace("_l2","")
            if "original_" in attack_label1:
                attack_label1 = attack_label1+"$^o$"
                attack_label1 = attack_label1.replace("original_","")
            if "dr_" in attack_label1:
                attack_label1 = attack_label1+"$^{dr}$"
                attack_label1 = attack_label1.replace("dr_","")
            if "fb_" in attack_label1:
                attack_label1 = attack_label1+"$^{fb}$"
                attack_label1 = attack_label1.replace("fb_","")
            if "ta_" in attack_label1:
                attack_label1 = attack_label1+"$^{ta}$"
                attack_label1 = attack_label1.replace("ta_","")
            if "ch_" in attack_label1:
                attack_label1 = attack_label1+"$^{ch}$"
                attack_label1 = attack_label1.replace("ch_","")
            if "art_" in attack_label1:
                attack_label1 = attack_label1+"$^{art}$"
                attack_label1 = attack_label1.replace("art_","")
            if "adv_lib_" in attack_label1:
                attack_label1 = attack_label1+"$^{al}$"
                attack_label1 = attack_label1.replace("adv_lib_","")

            for model in target_models:
                adv_distances = np.array(list(result[model][attack_label].values()))
                clip_num = (adv_distances > t_max_dist[i_ax]).sum()
                distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=t_max_dist[i_ax]),
                                                      return_counts=True)
                #counts[-1] = counts[-1] - int(clip_num*(t_max_dist[i]/max_bound[scenario.threat_model]))
                counts[-1] = counts[-1] - clip_num
                robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)
                area = np.trapz(robust_acc_clipped, distances_clipped)
                optimality = 1 - (area - t_best_area[i_ax]) / (t_clean_acc[i_ax] * t_max_dist[i_ax] - t_best_area[i_ax])
                if optimality<0:
                    optimality=0
                
                ax[i_ax].plot(distances_clipped, robust_acc_clipped, linewidth=1, linestyle='--',
                    label=f'{attack_label1}: {optimality:.2%}')
                i_ax += 1

        i_ax = 0
        for model in target_models:
            ax[i_ax].legend(loc='center right', labelspacing=.1, handletextpad=0.5)
            i_ax += 1

        fig_name = result_path / f'{scenario.threat_model}.pdf'
        fig.savefig(fig_name)
        i_ax=0
        for model in target_models:
            subpath = '/'+'cifar10/'+scenario.threat_model+'/'+scenario.threat_model+'_'+model+'.pdf'
            save_subfig(fig,ax[i_ax],args.dir,subpath)
            i_ax += 1
        plt.close()


    for key in threat_model_lst:
        np.savetxt(f"output_{key}.csv", Table[key], delimiter=",", fmt='%s')
        print(tabulate(Table[key], headers="firstrow", missingval="-", tablefmt="rst", floatfmt="0.3f"))
        print()
