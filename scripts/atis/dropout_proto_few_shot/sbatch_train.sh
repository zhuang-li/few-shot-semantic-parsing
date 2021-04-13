#!/bin/bash
#SBATCH --job-name=cross_template
# To set a project account for credit charging,
#SBATCH --account=fz25


# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Request for GPU,

#SBATCH --gres=gpu:1

#SBATCH --partition=m3c

# Memory usage (MB)
#SBATCH --mem=16000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=0-15:00:00



# To receive an email when job completes or fails
#SBATCH --mail-user=zhuang.li@monash.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=cross_template_atis_proto-%j.out

# Set the file for error log (stderr)
#SBATCH --error=cross_template_atis_proto-%j.err


# Command to run a gpu job
# For example:
module load cuda
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate
#module load pytorch


set -e

seed=0
dropout=0.5
dropout_i=0
proto_dropout=0.5
hidden_size=256
embed_size=200
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
sup_proto_turnover=2
ls=0.1
att_reg=1
relax_factor=10
lstm='lstm'
forward_pass=10
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_few_shot'
suffix=''
shuffle=0
k_shot_num=1
while getopts "d:r:h:e:a:l:f:o:k:p:m:t:v:" opt; do
  case $opt in
    d)
      dropout=$OPTARG
      echo "dropout is $dropout"
      ;;
    r)
      dropout_i=$OPTARG
      echo "dropout_i is $dropout_i"
      ;;
    h)
      hidden_size=$OPTARG
      echo "hidden_size is $hidden_size"
      ;;
    e)
      embed_size=$OPTARG
      echo "embed_size is $embed_size"
      ;;
    a)
      shuffle=$OPTARG
      echo "shuffle number is $shuffle"
      ;;
    l)
      lr=$OPTARG
      echo "lr is $lr"
      ;;
     f)
      k_shot_num=$OPTARG
      echo "shot k num is $k_shot_num"
      ;;
     o)
      optimizer=$OPTARG
      echo "optimizer is $optimizer"
      ;;
     k)
      lr_decay=$OPTARG
      echo "lr_decay is $lr_decay"
      ;;
     p)
      lr_decay_after_epoch=$OPTARG
      echo "lr_decay_after_epoch is $lr_decay_after_epoch"
      ;;
     m)
      max_epoch=$OPTARG
      echo "max_epoch is $max_epoch"
      ;;
     t)
      proto_dropout=$OPTARG
      echo "proto dropout is $proto_dropout"
      ;;
    v)
      suffix=$OPTARG
      echo "suffix is $suffix"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
data_dir="datasets/atis/query_split/few_shot_split_random_5_predishuffle_${shuffle}_shot_${k_shot_num}/"
model_dir="saved_models/atis/query_split/few_shot_split_random_5_predi/shuffle_${shuffle}_shot_${k_shot_num}/"
vocab=${data_dir}"train_vocab.bin"
train_file=${data_dir}"train.bin"

pre_model_name=model.atis.pre_train.${lstm}.hid${hidden_size}.embed${embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).att_reg${att_reg}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.p_dropout${proto_dropout}.k_shot${k_shot_num}.shuffle${shuffle}.suffix${suffix}

python -u dropout_few_shot_exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode pre_train \
    --lang atis_lambda \
    --n_way ${n_way} \
    --k_shot ${k_shot} \
    --att_reg ${att_reg} \
    --sup_proto_turnover ${sup_proto_turnover} \
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
    --save_to ${model_dir}${pre_model_name}\
    --glove_embed_path embedding/glove/glove.6B.200d.txt \
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file datasets/geo/lambda/question_split/test.bin \
. scripts/atis/dropout_proto_few_shot/fine_tune.sh ${model_dir}${pre_model_name}.bin
#2>>logs/geo/lambda/question_split/${model_name}.log
