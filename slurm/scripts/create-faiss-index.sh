#!/bin/bash
#SBATCH  -J vectordb
#SBATCH  -t 10-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h200:1

##SBATCH  --constraint=80g

##SBATCH -q cair

#SBATCH -q shamout

##SBATCH  -q nvidia-xxl



# singularity overlay storage
OVERLAY=/scratch/sas10092/ehr-foundation/overlay-512000M-15000K.ext3
SIF=/share/apps/admin/singularity-images/centos-8.2.2004.sif

singularity exec --nv --overlay "${OVERLAY}" "${SIF}" bash -lc "
  source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
  conda activate ehr-ragp
  set -x
  cd /scratch/sas10092/ehr-foundation
  python create_vdb_idx.py \
  --data-idx-path ./resources/downstream_index.parquet \
  --hf-dataset-path ./data/meds_normalized_arrow/ \
  --tokenizer-path ./resources/vocab.json \
  --ckpt-path ./models/pretraining/wandb/run-20260810_094728-roformer_transformer_17152740_1024_128_15_maskprob_12.5overlap/files/ckpt/roformer.ckpt \
  --embedder-model roformer \
  --use-type \
  --use-visit \
  --use-stage \
  --seq-length-q 1024 \
  --overlap-q 0 \
  --storage-path /faiss
"


# # on disk storage 
# conda activate ehr-ragp
# set -x

# python create_vdb_idx.py \
#   --data-idx-path ./resources/downstream_index.parquet \
#   --hf-dataset-path ./data/meds_normalized_arrow/ \
#   --tokenizer-path ./resources/vocab.json \
#   --ckpt-path ./models/pretraining/wandb/run-20260810_094728-roformer_transformer_17152740_1024_128_15_maskprob_12.5overlap/files/ckpt/roformer.ckpt \
#   --embedder-model roformer \
#   --use-type \
#   --use-visit \
#   --use-stage \
#   --seq-length-q 1024 \
#   --overlap-q 0 \
#   --storage-path /path/to/storage/directory/

