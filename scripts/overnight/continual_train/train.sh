#!/bin/bash
set -e

domains=('basketball' 'blocks' 'calendar' 'housing' 'publications' 'recipes' 'restaurants' 'socialnetwork')
data_dir="datasets/overnight/continual_split/"
model_dir="saved_models/overnight/continual_split/"
decode_dir="datasets/overnight/decode_results/"
permu_seed=$1 # seed for permutations
mode="train"
para_seed=0
subselect=0
ada_emr=0
dropout=0
dropout_i=0
hidden_size=256
num_exemplars_per_class=5
num_exemplars_per_task=50
num_exemplars_ratio=0.01
num_known_domains=1
embed_size=200
replay_dropout=1
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=50
max_epoch=10
patience=1000   # disable patience since we don't have dev set
beam_size=1
rebalance=0
augment=0
batch_size=64
lr=0.0025
alpha=0.95
ls=0.1
reg=0.5
ewc=0
sample_method='graph_clustering' # greedy_uniform graph_clustering
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='gss_emr'
lang='overnight_lambda'
suffix='sup'
attention='dot'
evaluator='denotation_evaluator' # smatch_evaluator default_evaluator denotation_evaluator
model_name=model.overn.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glo.ls${ls}.seed${seed}.cgm${clip_grad_mode}.p${parser}.sa_me${sample_method}

python -u life_exp.py \
    --use_cuda \
    --seed ${permu_seed} \
    --p_seed ${para_seed} \
    --lang ${lang} \
    --mode ${mode} \
    --batch_size ${batch_size} \
    --train_file ${data_dir} \
    --test_file ${data_dir} \
    --vocab ${data_dir} \
    --lstm ${lstm} \
    --ewc ${ewc} \
    --subselect ${subselect} \
    --attention ${attention} \
    --augment ${augment} \
    --ada_emr ${ada_emr} \
    --evaluator ${evaluator} \
    --reg ${reg} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --num_exemplars_per_task ${num_exemplars_per_task} \
    --num_exemplars_ratio ${num_exemplars_ratio} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --max_epoch ${max_epoch} \
    --rebalance ${rebalance} \
    --sample_method ${sample_method} \
    --lr ${lr} \
    --replay_dropout ${replay_dropout} \
    --parser ${parser} \
    --alpha ${alpha} \
    --decay_lr_every_epoch \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --clip_grad_mode ${clip_grad_mode} \
    --optimizer ${optimizer} \
    --lr_decay ${lr_decay} \
    --patience ${patience} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 20 \
    --log_every 50 \
    --save_to ${model_dir}${model_name} \
    --domains ${domains[@]} \
    --num_exemplars_per_class ${num_exemplars_per_class} \
    --num_known_domains ${num_known_domains} \
    --save_decode_to ${decode_dir} \
    --glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file ${data_dir} \

