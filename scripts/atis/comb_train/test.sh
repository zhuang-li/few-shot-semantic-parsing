#!/bin/bash
model_name=$(basename $1)
shuffle=$2
k_shot_num=$3
white_list=$4
parser='irnet'
evaluator="default_evaluator"
align_att=60
if [[ $white_list -eq 0 ]]
then
  data_dir_prefix="datasets/atis/query_split/supervised/few_shot_split_random_5_predi/"
elif [[ $white_list -eq 1 ]]
then
  data_dir_prefix="datasets/atis/query_split/supervised/no_use_white_list/few_shot_split_random_5_predi/"
fi
data_dir=${data_dir_prefix}"shuffle_${shuffle}_shot_${k_shot_num}/"
python sup_exp.py \
    --use_cuda \
    --mode test \
    --lang atis_lambda \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser} \
    --relax_factor 10 \
    --evaluator ${evaluator} \
    --test_file ${data_dir}query.bin \
    --clip_grad_mode norm \
    --align_att ${align_att} \
    --decode_max_time_step 100
