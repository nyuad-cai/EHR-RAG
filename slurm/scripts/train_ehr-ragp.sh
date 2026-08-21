#!/bin/bash
#SBATCH  -J 1y-mort
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

#SBATCH  --gres=gpu:h200:1

##SBATCH  --gres=gpu:h100:1

##SBATCH  --gres=gpu:a100:1
##SBATCH --constraint=80g

##SBATCH -q shamout
##SBATCH -q nvidia-xxl
#SBATCH -q cair



OVERLAY=/scratch/sas10092/ehr-foundation/overlay-512000M-15000K.ext3
SIF=/share/apps/admin/singularity-images/centos-8.2.2004.sif



# common settings 
BENCHMARK=mimic
VERSION=with-rtrieval
PROJECT_NAME=ehr-ragp-tuning
LOG_DIR=./models/hparams
TOKENIZER_PATH=./resources/vocab.json
DATA_PATH=./data/meds_normalized_arrow 
DATA_IDX_PATH=./resources/downstream_index.parquet
RUN_MODE=hparams

CHUNKING_STRATEGY=overlap
SPAN=256
USE_PROTOTYPES=0

# task
TASK=y_icu_readmit_30
MAIN_WINDOW_QUERY=within_stay_query
MAIN_WINDOW_HISTORY=within_stay_hist_full

TASK=y_mort
MAIN_WINDOW_QUERY=within48_query
MAIN_WINDOW_HISTORY=within48_hist_full

TASK=y_los_7
MAIN_WINDOW_QUERY=within24_query
MAIN_WINDOW_HISTORY=within24_hist_full

TASK=y_mort_12mo
MAIN_WINDOW_QUERY=within_stay_query
MAIN_WINDOW_HISTORY=within_stay_hist_full


CKPT=./models/pretraining/wandb/run-20260810_094728-roformer_transformer_17152740_1024_128_15_maskprob_12.5overlap/files/ckpt/roformer.ckpt
SEQ_LENGTH_Q=1024
OVERLAP_Q=0
singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" bash -lc "
  source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
  conda activate med-ehr
  set -x
  cd /scratch/sas10092/ehr-foundation
  torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 train_ehr_ragp.py \
    --backbone-name roformer \
    --job-id ${SLURM_JOB_ID} \
    --version ${VERSION} \
    --wandb-api-key $(cat ~/.wandb_token) \
    --wandb-log-dir ${LOG_DIR} \
    --task ${TASK} \
    --data-idx-path ${DATA_IDX_PATH} \
    --data-path ${DATA_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --seq-length-q ${SEQ_LENGTH_Q} \
    --overlap-q ${OVERLAP_Q} \
    --main-window-query ${MAIN_WINDOW_QUERY} \
    --main-window-history ${MAIN_WINDOW_HISTORY} \
    --ckpt-path ${CKPT} \
    --chunking-strategy ${CHUNKING_STRATEGY} \
    --benchmark ${BENCHMARK} \
    --span ${SPAN} \
    --project \
    $( [ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" )
"
