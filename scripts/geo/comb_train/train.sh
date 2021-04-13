#!/bin/bash
#SBATCH --job-name=cross_template
# To set a project account for credit charging,
#SBATCH --account=fz25


# Request CPU resource for a serial job
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4

# Request for GPU,
#SBATCH --gres=gpu:1

#SBATCH --partition=m3g

# Memory usage (MB)
#SBATCH --mem=32000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=01:00:00


# To receive an email when job completes or fails
#SBATCH --mail-user=zhuang.li@monash.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=geo-%j.out

# Set the file for error log (stderr)
#SBATCH --error=geo-%j.err


# Command to run a gpu job
# For example:
module load cuda
module load cmake
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate
#module load pytorch

set -e

shuffle=$1
k_shot_num=$2
white_list=$3
if [[ $white_list -eq 0 ]]
then
  data_dir_prefix="datasets/geo/lambda/query_split/supervised/few_shot_split_random_5_predi/"
  model_dir_prefix="saved_models/geo/lambda/query_split/supervised/few_shot_split_random_5_predi/"
elif [[ $white_list -eq 1 ]]
then
  data_dir_prefix="datasets/geo/lambda/query_split/supervised/no_use_white_list/few_shot_split_random_5_predi/"
  model_dir_prefix="saved_models/geo/lambda/query_split/supervised/no_use_white_list/few_shot_split_random_5_predi/"
fi
vocab=${data_dir_prefix}"shuffle_${shuffle}_shot_${k_shot_num}/support_vocab.bin"
train_file=${data_dir_prefix}"shuffle_${shuffle}_shot_${k_shot_num}/train.bin"
support_file=${data_dir_prefix}"shuffle_${shuffle}_shot_${k_shot_num}/support.bin"
model_dir=${model_dir_prefix}"shuffle_${shuffle}_shot_${k_shot_num}/"

augment=$4
discriminate_loss=0.4
align_att=60
dropout=0.5
dropout_i=0
hidden_size=256
embed_size=200
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=100
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=64
lr=0.0025
ls=0.1
clip_grad_mode='norm'
lstm='lstm'
optimizer='Adam'
parser='irnet'
suffix='dis_comb'
model_name=model.geo.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}

python -u sup_exp.py \
    --use_cuda \
    --seed 0 \
    --mode train \
    --augment ${augment} \
    --lang geo_lambda \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --support_file ${support_file} \
    --vocab ${vocab} \
    --align_att ${align_att} \
    --parser ${parser} \
    --discriminate_loss ${discriminate_loss} \
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
    --embed_fixed \
    --glove_embed_path embedding/glove/glove.6B.300d.txt \
    #--dev_file datasets/jobs/test.bin \
    #2>logs/geo/question_split/question_split/${model_name}.log

. scripts/geo/comb_train/test.sh ${model_dir}${model_name}.bin ${shuffle} ${k_shot_num} ${white_list}
echo "shuffle_"${shuffle}"_shot_"${k_shot_num}"_white_list_"${white_list}"_augment_"${augment}
#2>>logs/jobs/${model_name}.log
