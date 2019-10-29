#!/bin/bash

#params:
param_dataset=wadi
param_dataset_path=/home/caizhuo/research/anomaly/data/
param_mdi_method=gaussian_globalcov # ['gaussian', 'gaussian_globalcov', 'll']
#param_dataset_filename="_px.npy"
param_use_bnaf=0 # [1 for use, 0 for not use]
param_use_score="a.npy"
# run mdi

#### p1
python -W ignore cai_maxdiv.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p1.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
#          --use_score $param_use_score

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p1.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf \
          --output_score _p1

######## p2
python -W ignore cai_maxdiv.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p2.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
#          --use_score $param_use_score

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p2.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf \
          --output_score _p2


############# p3
python -W ignore cai_maxdiv.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p3.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
#          --use_score $param_use_score

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _p3.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf \
          --output_score _p3


######### px
python -W ignore cai_maxdiv.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _px.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf
#          --use_score $param_use_score

# run eval

python -W ignore cai_eval.py --dataset $param_dataset \
          --dataset_path $param_dataset_path \
          --dataset_filename _px.npy \
          --mdi_method $param_mdi_method \
          --use_bnaf $param_use_bnaf \
          --output_score _px


