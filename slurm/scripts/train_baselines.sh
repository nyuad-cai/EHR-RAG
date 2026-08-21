#!/bin/bash
#SBATCH  -J hparams
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1
#SBATCH  --constraint=80g

#SBATCH -q cair

##SBATCH -q shamout

##SBATCH  -q nvidia-xxl


eval "$(conda shell.bash hook)"
conda activate ehr-ragp

set -x

# common settings 
VERSION=baselines
PROJECT_NAME=ehr-ragp-tuning
LOG_DIR=./models/hparams
TOKENIZER_PATH=./resources/vocab.json
DATA_PATH=./data/meds_normalized_arrow
DATA_PATH_TEXT=./data/desc_gen_dataset
DATA_IDX_PATH=./resources/downstream_index.parquet
RUN_MODE=hparams

# task
TASK=y_icu_readmit_30
MAIN_WINDOW=within_stay_query
MAIN_WINDOW_DESCEMB=within_stay_descemb
MAIN_WINDOW_GENHPF=within_stay_genhpf
MAIN_WINDOW_REMED=within_stay_remed
TIME_FIELD=within_stay_remed_time
TIME_DIFF_FIELD=within_stay_remed_time_diff
TIME=48

# TASK=y_mort
# MAIN_WINDOW=within48_query
# MAIN_WINDOW_DESCEMB=within48_descemb
# MAIN_WINDOW_GENHPF=within48_genhpf
# MAIN_WINDOW_REMED=within48_remed
# TIME_FIELD=within48_remed_time
# TIME_DIFF_FIELD=within48_remed_time_diff
# TIME=48

# TASK=y_los_7
# MAIN_WINDOW=within24_query
# MAIN_WINDOW_DESCEMB=within24_descemb
# MAIN_WINDOW_GENHPF=within24_genhpf
# MAIN_WINDOW_REMED=within24_remed
# TIME_FIELD=within24_remed_time
# TIME_DIFF_FIELD=within24_remed_time_diff
# TIME=24

# TASK=y_mort_12mo
# MAIN_WINDOW=within_stay_query
# MAIN_WINDOW_DESCEMB=within_stay_descemb
# MAIN_WINDOW_GENHPF=within_stay_genhpf
# MAIN_WINDOW_REMED=within_stay_remed
# TIME_FIELD=within_stay_remed_time 
# TIME_DIFF_FIELD=within_stay_remed_time_diff 
# TIME=48



# # roberta
# CKPT=./models/pretraining/wandb/run-20260810_093926-roberta_transformer_17152730_512_64_15_maskprob_12.5overlap/files/ckpt/roberta.ckpt
# SEQ_LEN=512
# OVERLAP=0
# BATCH_SIZE=64
# python train_baselines.py --backbone-name roberta --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --run-mode $RUN_MODE #--use-time --use-numeric --use-stage --use-visit --use-type

# # longformer
# CKPT=./models/pretraining/wandb/run-20260810_094143-longformer_transformer_17152732_1024_128_15_maskprob_12.5overlap/files/ckpt/longformer.ckpt
# SEQ_LEN=1024
# OVERLAP=0
# BATCH_SIZE=32
# python train_baselines.py --backbone-name longformer --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE  --project-name $PROJECT_NAME --run-mode $RUN_MODE #--use-time --use-numeric --use-stage --use-visit --use-type

# # big_bird
# CKPT=./models/pretraining/wandb/run-20260810_094528-big_bird_transformer_17152739_1024_128_15_maskprob_12.5overlap/files/ckpt/bigbird.ckpt
# SEQ_LEN=1024
# OVERLAP=0
# BATCH_SIZE=16
# python train_baselines.py --backbone-name big_bird --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --run-mode $RUN_MODE #--use-time --use-numeric --use-stage --use-visit --use-type

# # roformer
# CKPT=./models/pretraining/wandb/run-20260810_094728-roformer_transformer_17152740_1024_128_15_maskprob_12.5overlap/files/ckpt/roformer.ckpt
# SEQ_LEN=1024
# OVERLAP=0
# BATCH_SIZE=16
# python train_baselines.py --backbone-name roformer --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --run-mode $RUN_MODE --use-time --use-numeric --use-stage --use-visit --use-type

# # modernbert long
# CKPT=./models/pretraining/wandb/run-20260810_100601-modernbert_transformer_17152752_1024_128_15_maskprob_12.5overlap/files/ckpt/modernbert.ckpt
# SEQ_LEN=4096
# OVERLAP=0
# BATCH_SIZE=24
# python train_baselines.py --backbone-name modernbert --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --run-mode $RUN_MODE # --use-time --use-numeric --use-stage --use-visit --use-type 

# # descemb cls-ft
# BATCH_SIZE=64
# python train_baselines.py --backbone-name descemb --job-id $SLURM_JOB_ID --data-path $DATA_PATH_TEXT --data-idx-path $DATA_IDX_PATH --wandb-log-dir $LOG_DIR --task $TASK \
#   --main-window $MAIN_WINDOW_DESCEMB --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --freeze \
#   --run-mode $RUN_MODE

# descemb bert-ft
BATCH_SIZE=64
python train_baselines.py --backbone-name descemb --job-id $SLURM_JOB_ID --data-path $DATA_PATH_TEXT --data-idx-path $DATA_IDX_PATH --wandb-log-dir $LOG_DIR \
  --task $TASK --main-window $MAIN_WINDOW_DESCEMB --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --batch-size $BATCH_SIZE --project-name $PROJECT_NAME \
  --run-mode $RUN_MODE

# # genhpf
# CKPT=./models/pretraining/wandb/run-20260816_144424-genhpf_simclr_genhpf_17259460_510_0_15_maskprob_12.5overlap/files/ckpt/genhpf.ckpt
# BATCH_SIZE=64
# python train_baselines.py --backbone-name genhpf --job-id $SLURM_JOB_ID --data-path $DATA_PATH_TEXT --data-idx-path $DATA_IDX_PATH  --wandb-log-dir $LOG_DIR --task $TASK \
#   --main-window $MAIN_WINDOW_GENHPF --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT  --batch-size $BATCH_SIZE --project-name $PROJECT_NAME \
#   --run-mode $RUN_MODE

# # remed
# CKPT=./models/pretraining/wandb/run-20260816_144424-genhpf_simclr_genhpf_17259460_510_0_15_maskprob_12.5overlap/files/ckpt/genhpf.ckpt
# BATCH_SIZE=64
# python train_baselines.py --backbone-name remed --job-id $SLURM_JOB_ID --data-path $DATA_PATH_TEXT --data-idx-path $DATA_IDX_PATH  --wandb-log-dir $LOG_DIR --task $TASK \
#   --main-window $MAIN_WINDOW_REMED --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT  --batch-size $BATCH_SIZE --project-name $PROJECT_NAME \
#   --time-field $TIME_FIELD --time-diff-field $TIME_DIFF_FIELD --pred-time $TIME --run-mode $RUN_MODE

# # hibehrt
# CKPT=./models/pretraining/wandb/run-20260810_093214-bert_hibehrt_17152724_512_64_15_maskprob_12.5overlap/files/ckpt/hibehrt.ckpt
# SEQ_LEN=1023
# OVERLAP=0
# BATCH_SIZE=64
# python train_baselines.py --backbone-name bert --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --variant hibehrt --run-mode $RUN_MODE --use-visit #--use-time --use-numeric --use-stage  --use-type  

# # medbert
# CKPT=./models/pretraining/wandb/run-20260810_092354-bert_medbert_17152697_512_64_15_maskprob_12.5overlap/files/ckpt/med-bert.ckpt
# SEQ_LEN=512
# OVERLAP=0
# BATCH_SIZE=64
# python train_baselines.py --backbone-name bert --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME  --variant medbert --use-visit --run-mode $RUN_MODE #--use-time --use-numeric --use-stage  --use-type 

# # behrt
# CKPT=./models/pretraining/wandb/run-20260810_092912-bert_behrt_17152720_512_64_15_maskprob_12.5overlap/files/ckpt/behrt.ckpt
# SEQ_LEN=512
# OVERLAP=0
# BATCH_SIZE=64
# python train_baselines.py --backbone-name bert --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --variant behrt --use-visit --run-mode $RUN_MODE #--use-time --use-numeric --use-stage  --use-type  

# # cehrbert
# CKPT=./models/pretraining/wandb/run-20260810_092628-bert_cehrbert_17152700_512_64_15_maskprob_12.5overlap/files/ckpt/cehrbert.ckpt
# SEQ_LEN=512
# OVERLAP=0
# BATCH_SIZE=64
# python train_baselines.py --backbone-name bert --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --variant cehrbert --use-time --use-numeric  --use-visit --use-stage --run-mode $RUN_MODE #--use-type 


# # ehrmamba
# CKPT=./
# SEQ_LEN=2048
# OVERLAP=0
# BATCH_SIZE=16
# python train_baselines.py --backbone-name mamba --job-id $SLURM_JOB_ID --data-path $DATA_PATH --data-idx-path $DATA_IDX_PATH --tokenizer-path $TOKENIZER_PATH \
#   --wandb-log-dir $LOG_DIR --task $TASK --main-window $MAIN_WINDOW --version $VERSION --wandb-api-key $(cat ~/.wandb_token) --ckpt-path $CKPT --seq-length $SEQ_LEN \
#   --seq-overlap $OVERLAP --batch-size $BATCH_SIZE --project-name $PROJECT_NAME --run-mode $RUN_MODE--use-time --use-numeric --use-stage --use-visit --use-type