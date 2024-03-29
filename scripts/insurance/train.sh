#!/bin/bash


data_dir_prefix="datasets/insurance/"
model_dir_prefix="saved_models/insurance/"
augment=0
vocab=${data_dir_prefix}"vocab.freq2.bin"
train_file=${data_dir_prefix}"train.bin"
model_dir=${model_dir_prefix}
discriminate_loss=0.4
align_att=60
dropout=0.5
dropout_i=0
hidden_size=256
embed_size=200
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=50
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=64
lr=0.0025
ls=0.1
clip_grad_mode='norm'
lstm='lstm'
optimizer='Adam'
parser='seq2seq'
suffix='insurance'
model_name=model.insurance.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}

python -u sup_exp.py \
    --use_cuda \
    --seed 0 \
    --mode train \
    --lang insurance_lambda \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --parser ${parser} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --label_smoothing ${ls} \
    --att_reg 0 \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --patience ${patience} \
    --optimizer ${optimizer} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --clip_grad_mode ${clip_grad_mode} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 40 \
    --log_every 50 \
    --save_to ${model_dir}${model_name} \
    --glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--sup_attention \
    #--dev_file datasets/jobs/test.bin \
    #2>logs/geo/question_split/question_split/${model_name}.log


./scripts/insurance/test.sh ${model_dir}${model_name}.bin


