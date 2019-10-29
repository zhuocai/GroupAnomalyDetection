#!/bin/bash

#params:
param_dataset=wadi
param_dataset_path=/home/caizhuo/research/anomaly/data/
param_mdi_method=gaussian_globalcov # ['gaussian', 'gaussian_globalcov', 'll']
param_dataset_filename="p.npy"
param_use_bnaf=0 # [1 for use, 0 for not use]
param_use_score="a.npy"
# run mdi

python -W ignore cai_maxdiv_multi.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename $param_dataset_filename \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
#          --use_score $param_use_score

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename $param_dataset_filename \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
