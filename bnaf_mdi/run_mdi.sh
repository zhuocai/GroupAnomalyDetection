#!/bin/bash

#params:
param_dataset=wadi
param_dataset_path=/home/caizhuo/research/anomaly/data/
param_mdi_method=gaussian_globalcov # ['gaussian', 'gaussian_globalcov']
param_dataset_filename="_np3.npy"
param_use_bnaf=1 # [1 for use, 0 for not use]

# run mdi

python -W ignore cai_maxdiv.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename $param_dataset_filename \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename $param_dataset_filename \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
