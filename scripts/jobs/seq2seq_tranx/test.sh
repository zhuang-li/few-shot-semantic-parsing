#!/bin/bash
model_name=$(basename $1)
shuffle=$2
k_shot_num=$3
evaluator="default_evaluator"
parser='ratsql'
data_dir="datasets/jobs/"
python sup_exp.py \
    --use_cuda \
    --mode test \
    --lang job_prolog \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser} \
    --relax_factor 10 \
    --test_file ${data_dir}test.bin \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --decode_max_time_step 100
