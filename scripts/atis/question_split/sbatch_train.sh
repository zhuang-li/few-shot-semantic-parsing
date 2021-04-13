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

#SBATCH --partition=m3g

# Memory usage (MB)
#SBATCH --mem=16000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=0-10:00:00


# To receive an email when job completes or fails
#SBATCH --mail-user=zhuang.li@monash.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=cross_template_atis-%j.out

# Set the file for error log (stderr)
#SBATCH --error=cross_template_atis-%j.err


# Command to run a gpu job
# For example:
module load cuda
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate

set -e

data_dir="datasets/atis/question_split/"
model_dir="saved_models/atis/question_split/"
seed=0
vocab=${data_dir}"vocab.freq2.bin"
train_file=${data_dir}"train.bin"
dropout=0.5
dropout_i=0
hidden_size=256
embed_size=128
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=200
patience=1000   # disable patience since we don't have dev set
beam_size=1
batch_size=64
lr=0.0025
alpha=0.95
ls=0.1
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_complete'
suffix=''
attention='dot'

while getopts "d:r:h:e:a:l:f:o:k:p:m:" opt; do
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
      action_embed_size=$action_embed_size
      echo "action_embed_size is $action_embed_size"
      ;;
    l)
      lr=$OPTARG
      echo "lr is $lr"
      ;;
     f)
      alpha=$OPTARG
      echo "alpha is $alpha"
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

model_name=model.atis.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}

python -u exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --attention ${attention} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
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
    --decode_max_time_step 50 \
    --log_every 50 \
    --save_to ${model_dir}${model_name} \
    --dev_file ${data_dir}dev.bin \
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file ${data_dir}test.bin \
. scripts/atis/question_split/test.sh ${model_dir}${model_name}.bin
#2>>logs/geo/lambda/question_split/${model_name}.log