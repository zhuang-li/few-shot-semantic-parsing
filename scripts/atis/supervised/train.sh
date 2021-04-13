#!/bin/bash
set -e

seed=0

shuffle=$1
k_shot_num=$2
suffix=$3
data_dir="datasets/atis/query_split/supervised/few_shot_split_random_5_predi/shuffle_${shuffle}_shot_${k_shot_num}/"
model_dir="saved_models/atis/query_split/supervised/few_shot_split_random_5_predi/shuffle_${shuffle}_shot_${k_shot_num}/"

vocab=${data_dir}"train_vocab.bin"
train_file=${data_dir}"train.bin"

dropout=0.5
dropout_i=0
proto_dropout=0.5  # the proto embedding rate is 1 - proto_dropout
att_reg=0
hidden_size=256
embed_size=300
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=100
patience=1000   # disable patience since we don't have dev set
batch_size=64
beam_size=1
n_way=30
k_shot=1
sup_proto_turnover=2
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
parser='irnet'
pre_model_name=model.atis.pre_train.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.att_reg${att_reg}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.sup_turn${sup_proto_turnover}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.p_dropout${proto_dropout}.parser${parser}.suffix${suffix}

python -u exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode train \
    --lang atis_lambda \
    --n_way ${n_way} \
    --k_shot ${k_shot} \
    --att_reg ${att_reg} \
    --query_num ${query_num} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --valid_every_epoch 1 \
    --forward_pass ${forward_pass} \
    --lstm ${lstm} \
    --sup_proto_turnover ${sup_proto_turnover} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --proto_dropout ${proto_dropout} \
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
    --log_every 50 \
    --save_to ${model_dir}${pre_model_name} \
    --glove_embed_path embedding/glove/glove.6B.300d.txt \
    #    --sup_attention \
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file datasets/geo/lambda/question_split/test.bin \
#. scripts/atis/dropout_proto_few_shot/fine_tune.sh ${model_dir}${pre_model_name}.bin ${shuffle} ${k_shot_num} ${suffix}
#2>>logs/geo/lambda/question_split/${model_name}.log
