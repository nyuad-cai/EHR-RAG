# EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records

<p align="center">
  <img src="assets/main-figure.png" width="95%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.12335">Paper</a> |
  <a href="https://github.com/nyuad-cai/EHR-RAGp">Code</a>
</p>

---

# Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [MEDS Conversion](#meds-conversion)
- [Data Preprocessing](#data-preprocessing)
- [Vector Index Setup](#vector-index-setup)
- [Patient Timeline Construction](#patient-timeline-construction)
- [Pretraining](#pretraining)
- [Baseline Training](#baseline-training)
- [EHR-RAGp Training](#ehr-ragp-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Reproducibility Notes](#reproducibility-notes)
- [Hardware Requirements](#hardware-requirements)
- [Configuration](#configuration)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

# Overview

Electronic Health Records (EHR) contain rich longitudinal patient information and are widely used in predictive modeling applications. However, effectively leveraging historical data remains challenging due to long trajectories, heterogeneous events, temporal irregularity, and the varying relevance of past clinical context.

EHR-RAGp is a retrieval-augmented framework for structured EHR data that dynamically retrieves and integrates clinically relevant patient history using a prototype-guided retrieval mechanism.

The framework:
- Constructs longitudinal patient trajectory databases
- Retrieves relevant historical segments
- Aligns retrievals using latent prototypes
- Fuses retrieved history with current patient state
- Supports multiple downstream clinical prediction tasks

---

# Key Features

- Retrieval-augmented EHR modeling framework
- Prototype-guided retrieval alignment
- Multi-granular chunking strategies
- Longitudinal patient trajectory modeling
- MEDS-based standardized EHR representation
- Compatible with existing EHR  models
- Fully reproducible MIMIC-IV pipeline

---

# Repository Structure

```bash
EHR-RAGp/
│
├── assets/                     # Figures and visualizations
├── configs/                    # Experiment configurations
├── data/                       # Processed MEDS data
├── preprocessing/              # Preprocessing scripts
├── models/                     # Model implementations
├── retrieval/                  # Retrieval modules
├── baselines/                  # Baseline implementations
├── experiments/                # Training/evaluation scripts
├── utils/                      # Utility functions
│
├── preprocess.py
├── pretrain.py
├── wo_hparams_opt.py
├── w_hparams_opt.py
├── final_eval.py
│
├── environment.yml
└── README.md
```

---

# Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ehr-ragp
```

---

# Dataset

All experiments are conducted using:

- **MIMIC-IV v3.1**
- Link: https://physionet.org/content/mimiciv/3.1/

Access requires:
1. PhysioNet account
2. CITI Program training: **Data or Specimens Only Research**
3. Data usage agreement signage

**Note:** Raw patient data are NOT distributed in this repository. Users have to download the raw MIMIC-IV v3.1 dataset on their premises.

---

# MEDS Conversion

We convert raw MIMIC-IV records into the standardized **Medical Event Data Standard (MEDS)** format.

This code works on top of already extracted MIMIC-IV in MEDS format.
    to use it you need to first extract the dataset in MEDS format using 
    [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) (V 0.1.2).
    post extract, move the MEDS_output directory to data directory. The preprocessing codes runs a stages,
    the next satge can not start unless the current one finishes.
---

# Data Preprocessing

After MEDS conversion, run the end-to-end preprocessing pipeline:

```bash
python preprocess.py
```

The script performs the full post-MEDS preprocessing workflow, including:

  1.  Remove HCPCS from patient timelines as they may constitute a tempral leakage (to be updated once resolved)
  2.  Convert OMR measurement from text_value into numeric_value
  3.  Segment patient timeline into (e.g., OUTPATIENT, ED, INPATIENT, ICU)
  4.  Assign visit id to each consecutive visit in the patient timeline (e.g., V1, V2,....)
  5.  Add time tokens between consecutive visits (e.g., TIME-GAP//1-YR, TIME-GAP//1-M)
  6.  Add token type anootation (e.g., MEDICATION, LAB_RESULT)
  7.  Eliminate outliers in numeric_value column
  8.  Eliminate rare event from vocab and timeline (thereshold >3)
  9.  Normalize numeric value column
  10. Event type collection and Tokenizer vocab buidling (vocab.json)
  11. Build pretraining index from train split (pretrain_idx.parquet)
  12. Build a readily tokenized full dataset (Arrow format)
  13. Downstream cohort filtering
  14. Ground truth labels extraction for downstreak tasks
  15. Query/History boundaries identification  



# Vector Index Setup

We use [Facebook AI Similarity Search (FAISS)](https://faiss.ai/index.html) to build vector indices. 

Run 
```bash
python create_vdb_idx.py
```
**Note:**

The file is prepared to create vector indices for all chunking strategies. As we create single index per patienet stay (1 stay = 1 index.faiss file), each training example will have its own index. We recommedn using singularity overlay if indices creation is being performed on HPC to avoid exceeding file quota limits.
# Pretraining

We pretrain the encoder using Masked Language Modeling (MLM).

Run pretraining:

```bash
export TOKENIZER_PATH="./resources/vocab.json"
export DATA_PATH="./data/meds_arrow"
export DATA_IDX_PATH="./resources/pretrain_idx.parquet"
export LOG_DIR="./models/mlm"
export VERSION="roformer"
export PRETRAIN_MODE="mlm"
export BACKBONE="roformer"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
torchrun --nproc_per_node=4 pretrain.py \
    --learning-rate 2.2908676527677725e-05 \
    --weight-decay 1e-2 \
    --max-epochs 100 \
    --batch-size 16 \
    --chunk-length 1024 \
    --overlap 128
```

**Main settings:**
- Backbone: RoFormer-base
- Context length: 1024
- MLM masking ratio: 15%
- Optimizer: AdamW
- GPUs: 4× NVIDIA A100


**Note:**
The code is compatable with huggingface transformers models and can be used to pretrain other different encoder based backbones. below we provide our pretrained model checkpoints via masked language modeling for:

<u>HuggingFace Models:</u>
1. [RoFormer Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/roformer-mlm-ehr.ckpt)
2. [ModernBERT Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/modernbert-mlm-ehr.ckpt)
3. [Longformer Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/longformer-mlm-ehr.ckpt)
4. [BigBird Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/bigbird-mlm-ehr.ckpt)
5. [RoBERTa Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/roberta-mlm-ehr.ckpt)
6. [BERT Checkpoint](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/bert-mlm-ehr.ckpt)


<u>Clinical Foundation Models:</u>
1. [MedBERT](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/medbert-mlm-ehr.ckpt)
1. [CEHR-BERT](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/cehrbert-mlm-ehr.ckpt)
1. [BEHRT](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/behrt-mlm-ehr.ckpt)
1. [HI-BEHRT](https://github.com/nyuad-cai/EHR-RAGp/releases/download/v0.1-checkpoints/hibehrt-mlm-ehr.ckpt)

---

# Baseline Training

## Hyperparameter Optimization

Run baseline tuning:

```bash
python wo_hparams_opt.py --config-path ./slurm/config/hparams/<task-name>/<baseline-name>.yaml 
```

Included baselines:
- Med-BERT
- BEHRT
- CEHR-BERT
- LongFormer
- BigBird
- RoFormer
- ModernBERT
- EHRMamba
- REMed

---

## final evaluation
After completing hyperparameters tuning, run final evaluation using the best set of hyperparameters using

```bash
python final_eval.py --config-path ./slurm/config/eval/1<task-name>/<baseline-name>.yaml 
```

# EHR-RAGp Training

## Hyperparameter Optimization

Run retrieval-augmented experiments:

```bash
python w_hparams_opt.py 
```

Key configurable components:
- Chunking strategy
- Number of prototypes
- Retrieval depth
- Prototype temperatures
- Fusion module

---

# Evaluation

Run downstream evaluation:

```bash
python final_eval.py
```

Supported tasks:
- In-hospital mortality
- ICU readmission
- Long length-of-stay
- 1-year mortality

Metrics:
- AUROC
- AUPRC
- Bootstrap confidence intervals

---

# Results

Main findings:
- EHR-RAGp consistently outperforms transformer and EHR foundation baselines
- Prototype-guided retrieval improves retrieval quality
- Retrieval augmentation improves existing EHR foundation models

---

# Reproducibility Notes

To ensure reproducibility:

- All splits are patient-level
- Test patients are excluded from pretraining
- Random seeds are fixed
- Hyperparameter search uses Bayesian optimization
- Confidence intervals computed with bootstrapping

Recommended seed:

```python
SEED = 24
```

---

# Hardware Requirements

Recommended:
- NVIDIA H100 / A100 GPUs
- ≥80GB GPU memory for full experiments

Minimum:
- Single high-memory GPU with reduced batch size

Approximate training cost:
- Pretraining: 4× A100 GPUs
- Downstream training: 1× H100

---

# Configuration

Example configuration:

```yaml
model:
  backbone: roformer
  hidden_dim: 768
  prototypes: 512

retrieval:
  top_m: 24
  chunk_size: 256

training:
  batch_size: 16
  lr: 1e-4
```

---

# Citation

```bibtex
@article{shurrab2026ehrragp,
  title={EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records},
  author={Shurrab, Saeed and Al-Omari, Mariam and El Samad, Dana and Shamout, Farah E.},
  journal={arXiv preprint arXiv:2605.12335},
  year={2026}
}
```

---

# Acknowledgements

This work builds upon:
- MIMIC-IV
- MEDS ecosystem
- PhysioNet
- RoFormer
- FAISS

We thank the MEDS contributors and the Clinical AI Lab at NYU Abu Dhabi.