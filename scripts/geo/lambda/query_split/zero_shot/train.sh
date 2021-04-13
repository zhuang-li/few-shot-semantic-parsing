#!/bin/bash
set -e

seed=${1:-0}
vocab="datasets/geo/lambda/query_split/few_shot_split_random_5_predi/align_threshold_0.5/train_vocab.bin"
train_file="datasets/geo/lambda/query_split/few_shot_split_random_5_predi/align_threshold_0.5/train.bin"
data_dir="datasets/geo/lambda/query_split/few_shot_split_random_5_predi/align_threshold_0.5/"
dropout=0.5
dropout_i=0
hidden_size=256
embed_size=128
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=200
patience=1000   # disable patience since we don't have dev set
batch_size=64
beam_size=1
n_way=30
k_shot=1
query_num=15
lr=0.0025
alpha=0.95
ls=0.1
relax_factor=10
lstm='lstm'
forward_pass=10
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_complete'
suffix='1'
pre_model_name=model.geo.pre_train.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.relax_factor${relax_factor}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}
fine_model_name=model.geo.fine_tune.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.relax_factor${relax_factor}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}

python -u maml_exp.py \
    --use_cuda \
    --seed ${seed} \
    --few_shot_mode pre_train \
    --lang geo_lambda \
    --n_way ${n_way} \
    --k_shot ${k_shot} \
    --query_num ${query_num} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --valid_every_epoch 1 \
    --forward_pass ${forward_pass} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --alpha ${alpha} \
    --parser ${parser} \
    --clip_grad_mode ${clip_grad_mode} \
    --optimizer ${optimizer} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --batch_size ${batch_size} \
    --beam_size ${beam_size} \
    --glorot_init \
    --patience ${patience} \
    --decode_max_time_step 40 \
    --log_every 10 \
    --save_to saved_models/geo/lambda/query_split/zero_shot/${pre_model_name}
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file datasets/geo/lambda/question_split/test.bin \
. scripts/geo/lambda/query_split/zero_shot/fine_tune.sh saved_models/geo/lambda/query_split/zero_shot/${pre_model_name}.bin
. scripts/geo/lambda/query_split/zero_shot/test.sh saved_models/geo/lambda/query_split/zero_shot/${fine_model_name}.bin
#2>>logs/geo/lambda/question_split/${model_name}.log