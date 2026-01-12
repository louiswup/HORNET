# **HORNET**: Fast and Minimal Adversarial Perturbations
Jiaping Wu, Antonio Emanuele Cinà, Francesco Villani, Zhaoqiang Xia, Luca Demetrio, Luca Oneto, Davide Anguita, Fabio Roli, Xiaoyi Feng

We propose HORNET, a novel framework for crafting minimum-norm transferable adversarial examples. HORNET integrates fixed-budget attacks with an efficient search strategy to minimize perturbation size while maintaining high transfer success rates.

> **ℹ️ Original Project:** [attackbench/AttackBench] by [Antonio Emanuele Cinà $^\star$, Jérôme Rony $^\star$, ecc] (<https://attackbench.github.io>).

## Requirements and Installations

- python==3.9
- sacred
- pytorch==1.12.1
- torchvision==0.13.1
- adversarial-robustness-toolbox
- foolbox
- torchattacks
- cleverhans
- deeprobust
- robustbench https://github.com/RobustBench/robustbench
- adv_lib https://github.com/jeromerony/adversarial-library

Clone the Repository:
```bash
git clone https://github.com/louiswup/HORNET.git
cd HORNET
```

Use the provided `environment.yml` file to create a Conda environment with the required dependencies:
```bash
conda env create -f environment.yml
```

Activate the Conda environment: 
```bash
conda activate attackbench-HORNET
```



## Usage

To run HORNET_art_fgm on C1 model and transfer to C2,C3,C4,C5 models, save the results in the "results_trans/cifar10" directory:

```bash
C1='standard'
C2='zhang_2020_small'
C3='stutz_2020'
C4='xiao_2020'
C5='wang_2023_small'
atk='art_fgm_hornet'
python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C1} model.target_models='['\"${C2}\"','\"${C3}\"','\"${C4}\"','\"${C5}\"']'
```

Command Breakdown:
- `-F results_dir/`: Specifies the directory results_dir/ where the attack results will be saved.
- `with`: Keyword for sacred.
- `attack.${atk}`: Indicates the use of the fgm attack from the adversarial-robustness-toolbox library.
- `model.{C1}`: Specifies the model 'standard' to be the source model.
- `model.target_models='['\"${C2}\"','\"${C3}\"','\"${C4}\"','\"${C5}\"']'`: Specifies the model 'zhang_2020_small', ecc to be the target models.

Other Optional Command:
- `attack.threat_model="l2"`: Sets the threat model to $\ell_2$, constraining adversarial perturbations based on the $\ell_2$ norm.
- `dataset.num_samples=1000`: Specifies the number of samples to use from the CIFAR-10 dataset during the attack.
- `dataset.batch_size=64`: Sets the batch size for processing the dataset during the attack.
- `seed=42`: Sets the random seed for reproducibility.

After the attack completes, you can find the results saved in the specified results_trans/ directory.

### How to Use the Attack

HORNET implements several attack methods (e.g., `art_fgm_hornet`, `art_apgd_hornet`) that can be run via a unified command-line interface. Below are the key steps for using the attack.
- You can extend or modify attacks by editing the configuration functions in attack_evaluation/attacks/[lib]/configs.py. Each attack requires a config function and a corresponding getter function.
- When running one experiment, use attack.${atk} to set the attack.

## compute and print tables
After get a couple of results then get the result tables and plot the security curves:

```bash
python -m analysis.print_tables_and_plot -d [results_save_dir] -dist actual --source_models [source_model1] [source_model2] ... --target_models [target_model1]  [target_model2] ...
```

Command Breakdown:
- `-d `:  the directory  where the attack results are saved.
- `-dist`: Indicates the distance type options are ['actual','best']
- `--source_models`: Specifies the source models in result tables and figures.
- `--target_models`: Specifies the target models in result tables and figures.

After the attack completes, you can find the results saved in pwd/output_l[0,1,2,inf].csv and in [results_save_dir]/[l0,l1,l2,linf]/*.pdf

## Attack format

To have a standard set of inputs and outputs for all the attacks, the wrappers for all the implementations (including libraries) must have the following format:

- inputs:
    - `model`: `nn.Module` taking inputs in the [0, 1] range and returning logits in $\mathbb{R}^K$
    - `inputs`: `FloatTensor` representing the input samples in the [0, 1] range
    - `labels`: `LongTensor` representing the labels of the samples
    - `targets`: `LongTensor` or `None` representing the targets associated to each samples
    - `targeted`: `bool` flag indicating if a targeted attack should be performed
- output:
    - `adv_inputs`: `FloatTensor` representing the perturbed inputs in the [0, 1] range
