#!/bin/bash
C1='standard'
C2='zhang_2020_small'
C3='stutz_2020'
C4='xiao_2020'
C5='wang_2023_small'
gpu=3
atk='art_fgm_hornet'
CUDA_VISIBLE_DEVICES=${gpu} python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C1} model.target_models='['\"${C2}\"','\"${C3}\"','\"${C4}\"','\"${C5}\"']' 
CUDA_VISIBLE_DEVICES=${gpu} python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C2} model.target_models='['\"${C1}\"','\"${C3}\"','\"${C4}\"','\"${C5}\"']'
CUDA_VISIBLE_DEVICES=${gpu} python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C3} model.target_models='['\"${C2}\"','\"${C1}\"','\"${C4}\"','\"${C5}\"']'
CUDA_VISIBLE_DEVICES=${gpu} python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C4} model.target_models='['\"${C2}\"','\"${C3}\"','\"${C1}\"','\"${C5}\"']'
CUDA_VISIBLE_DEVICES=${gpu} python -m attack_evaluation.run_with_trans -F results_trans with attack.${atk} model.${C5} model.target_models='['\"${C2}\"','\"${C3}\"','\"${C4}\"','\"${C1}\"']'
