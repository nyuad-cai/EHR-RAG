import os
import torch
import wandb
import optuna
import argparse

import lightning.pytorch as lt

from types import SimpleNamespace
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from src.models.models import EvalModel
from lightning.pytorch.loggers import WandbLogger
from src.models.baseline_models import HiBEHRTModule 
from lightning.pytorch.utilities import rank_zero_only
from src.models.baseline_models import DescEmbEvalModel
from src.data.baseline_datasets import HiBEHRTEvalCollator
from src.data.baseline_datasets import DescEmbDataset, DescEmbCollator
from src.models.baseline_models import GenHPFDownstreamModule, GenHPFEncoder
from src.models.baseline_models import REMedWithGenHPF, REMedLightningModule
from src.data.baseline_datasets import REMedGenHPFPoolDataset, REMedGenHPFCollator
from src.data.datasets import limits, SequencesGenerator, EvalDataset, EvalCollator
from src.data.baseline_datasets import HierarchicalGenHPFDataset, GenHPFEvalCollator
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos, load_config_with_env



parser = argparse.ArgumentParser(description='MLM pretraining command line interface')

# backbone / experiment
parser.add_argument("--backbone-name", type=str, default=None)
parser.add_argument("--variant", type=str, default=None)
parser.add_argument("--job-id", type=str, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--version", type=str, default=None)
parser.add_argument("--ckpt-path", type=str, default=None)
# logger
parser.add_argument("--wandb-api-key", type=str, default=None)
parser.add_argument("--wandb-log-dir", type=str, default=None)
parser.add_argument("--project-name", type=str, default=None)
# dataset
parser.add_argument("--main-window", type=str, default=None)
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--data-idx-path", type=str, default=None)
# sequence generation
parser.add_argument("--tokenizer-path", type=str, default=None)
parser.add_argument("--seq-length", type=int, default=None)
parser.add_argument("--seq-overlap", type=int, default=None) 
# dataloader
parser.add_argument("--batch-size", type=int, default=None)
# REMed-specific
parser.add_argument("--time-field", type=str, default=None)
parser.add_argument("--time-diff-field", type=str, default=None)
parser.add_argument("--pred-time", type=float, default=None)
# boolean option
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--use-time", action="store_true")
parser.add_argument("--use-numeric", action="store_true")
parser.add_argument("--use-stage", action="store_true")
parser.add_argument("--use-visit", action="store_true")
parser.add_argument("--use-type", action="store_true")
# run mode
parser.add_argument("--run-mode", type=str, default='hparams',choices=['hparams'])
args = parser.parse_args()


if args.backbone_name == 'descemb':
    mode = 'cls-ft' if args.freeze else 'bert-ft'
elif args.variant is not None:
    mode = args.variant
else:
    mode = 'base'

def get_run_dir(wandb_logger):
    run = wandb_logger.experiment
    d = getattr(run, "dir", None)
    if callable(d):
        d = d()
    if not d:
        base = wandb_logger.save_dir or "."
        d = os.path.join(base, str(wandb_logger.version))
    return d 

@rank_zero_only
def make_dir(p):
    os.makedirs(p, exist_ok=True)
    
if args.run_mode == 'hparams':
    def objective(trial: optuna.trial.Trial) -> float:
        
        try:
            wandb.login(key=args.wandb_api_key)

            wandb_logger = WandbLogger(project=args.project_name,
                                    save_dir=args.wandb_log_dir,
                                    version=f"{args.backbone_name}_{mode}_{args.job_id}_{args.task}_{args.version}_{trial.number}",
                                    name=f"{args.backbone_name}_{mode}_{args.job_id}_{args.task}_{args.version}_{trial.number}",
                                    tags=[args.version,args.task,args.backbone_name,mode]) 

            run_dir = get_run_dir(wandb_logger)         
            ckpt_dir = os.path.join(run_dir, "ckpt")
            make_dir(ckpt_dir)
            prediction_csv_path = os.path.join(ckpt_dir,"test_predictions.csv")

            if args.backbone_name == 'bert' and args.variant == 'hibehrt':
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant='hibehrt')
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}

                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                            chunk_length=args.seq_length,
                                            overlap=args.seq_overlap)

                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            limits_dict=limits,
                                            task=args.task,
                                            main_window=args.main_window,
                                            seq_length=args.seq_length + 1,
                                            use_numeric=args.use_numeric,
                                            add_cls=False,
                                            use_time=args.use_time,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train')
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            limits_dict=limits,
                                            task=args.task,
                                            main_window=args.main_window,
                                            seq_length=args.seq_length + 1,
                                            use_numeric=args.use_numeric,
                                            add_cls=False,
                                            use_time=args.use_time,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning')
                collate_fn = HiBEHRTEvalCollator(seq_gen=seq_gen,
                                                chunk_length=256,
                                                overlap=32,
                                                add_cls_per_chunk=True)
                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                
                model = HiBEHRTModule(config=cfg,
                                    backbone=ModelClass,
                                    ckpt_path=args.ckpt_path,
                                    lr=learning_rate,
                                    wd=weight_decay,
                                    max_epochs=75,
                                    freeze=False,
                                    pooling='mean',
                                    dropout=0.1,
                                    optimizer='sgd',
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    prediction_csv_path=prediction_csv_path)
                
            elif args.backbone_name in 'bert' and args.variant == 'medbert':
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                                chunk_length=args.seq_length,
                                                overlap=args.seq_overlap)
                collate_fn = EvalCollator()
                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train')
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning')
                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                model = EvalModel(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=75,
                                pooling='cls',
                                use_numeric=args.use_numeric,
                                use_time=args.use_time,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                freeze=False,
                                optimizer='sgd',
                                prediction_csv_path=prediction_csv_path)
                
            elif args.backbone_name in 'bert' and args.variant == 'behrt':
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                                chunk_length=args.seq_length,
                                                overlap=args.seq_overlap)
                collate_fn = EvalCollator()
                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train')
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning')
                
                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                model = EvalModel(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=75,
                                pooling='cls',
                                use_numeric=args.use_numeric,
                                use_time=args.use_time,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                freeze=False,
                                optimizer='sgd',
                                prediction_csv_path=prediction_csv_path)
                
            elif args.backbone_name in 'bert' and args.variant == 'cehrbert':
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                                chunk_length=args.seq_length,
                                                overlap=args.seq_overlap)
                collate_fn = EvalCollator()
                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train')
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning')
                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                model = EvalModel(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=75,
                                pooling='cls',
                                use_numeric=args.use_numeric,
                                use_time=args.use_time,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                freeze=False,
                                optimizer='sgd',
                                prediction_csv_path=prediction_csv_path)   

            elif args.backbone_name == 'mamba':
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=None)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                                chunk_length=args.seq_length,
                                                overlap=args.seq_overlap)
                collate_fn = EvalCollator()
                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train')
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning')
                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                model = EvalModel(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=75,
                                pooling='cls',
                                use_numeric=args.use_numeric,
                                use_time=args.use_time,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                freeze=False,
                                optimizer='sgd',
                                prediction_csv_path=prediction_csv_path)
            elif args.backbone_name in ['roberta','longformer','big_bird','roformer','modernbert']:
                ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=None)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                                chunk_length=args.seq_length,
                                                overlap=args.seq_overlap)
                collate_fn = EvalCollator()
                train_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='train',
                                            use_long_context= True if args.seq_length > 1024 else False)
                val_dataset = EvalDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            seq_gen=seq_gen,
                                            seq_length=args.seq_length,
                                            limits_dict=limits,
                                            main_window=args.main_window,
                                            task=args.task,
                                            use_time=args.use_time,
                                            use_numeric=args.use_numeric,
                                            use_type=args.use_type,
                                            use_visit=args.use_visit, 
                                            use_stage=args.use_stage,
                                            split='tuning',
                                            use_long_context= True if args.seq_length >1024 else False)

                cfg = ConfigClass(
                    vocab_size=seq_gen.tokenizer.vocab_size,
                    cls_token_id=seq_gen.tokenizer.cls_id,
                    pad_token_id=seq_gen.tokenizer.pad_id,
                    type_vocab_size=43,
                    visit_vocab_size=990,
                    stage_vocab_size=5,
                    refernece_compile=False)
                cfg = fix_roberta_longformer_max_pos(cfg)
                model = EvalModel(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                wd=weight_decay,
                                max_epochs=75,
                                pooling='cls',
                                use_numeric=args.use_numeric,
                                use_time=args.use_time,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                freeze=False,
                                optimizer='sgd',
                                prediction_csv_path=prediction_csv_path)
                
            elif args.backbone_name == 'descemb':
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
                dropout = trial.suggest_float("dropout", low=0.1, high=0.5, step=0.2)
                hparams={'learning_rate': learning_rate,
                        'dropout': dropout}
                train_dataset = DescEmbDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            task=args.task,
                                            main_window=args.main_window,
                                            max_word_len=12,
                                            max_events=510,
                                            split='train') 
                val_dataset = DescEmbDataset(dataset_path=args.data_path,
                                            data_idx_path=args.data_idx_path,
                                            task=args.task,
                                            main_window=args.main_window,
                                            max_word_len=12,
                                            max_events=510,
                                            split='tuning') 
                collate_fn = DescEmbCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
                cfg = SimpleNamespace(bert_model_name="google/bert_uncased_L-2_H-128_A-2",
                                    pred_embed_dim=128,
                                    pred_hidden_dim=256,     
                                    max_event_len=510,       
                                    rnn_layer=1,
                                    init_bert_random=False,  
                                    task="binary")
                model = DescEmbEvalModel(config=cfg,
                                        lr=learning_rate,
                                        max_epochs=75,
                                        dropout=dropout,
                                        freeze=args.freeze,
                                        prediction_csv_path=prediction_csv_path)
            elif args.backbone_name in ['genhpf']:
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
                dropout = trial.suggest_float("dropout", low=0.1, high=0.5, step=0.2)
                hparams={'learning_rate': learning_rate,
                        'dropout': dropout}
                train_dataset = HierarchicalGenHPFDataset(
                    dataset_path=args.data_path,
                    data_idx_path=args.data_idx_path,
                    seq_field=args.main_window,
                    label_field=args.task,
                    split='train',
                    tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                    max_events=511,
                    max_tokens=64)
                val_dataset = HierarchicalGenHPFDataset(
                    dataset_path=args.data_path,
                    data_idx_path=args.data_idx_path,
                    seq_field=args.main_window,
                    label_field=args.task,
                    split='tuning',
                    tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                    max_events=511,
                    max_tokens=64)
                collate_fn = GenHPFEvalCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
                encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                                        pad_token_id=train_dataset.tokenizer.pad_token_id,
                                        encoder_embed_dim=128,
                                        encoder_layers=2,
                                        encoder_ffn_embed_dim=512,
                                        encoder_attention_heads=4,
                                        agg_embed_dim=128,
                                        agg_layers=4,
                                        agg_ffn_embed_dim=512,
                                        agg_attention_heads=4,
                                        dropout=dropout,
                                        max_token_len=64,   
                                        max_events=511,
                                        encoder_only=False,
                                        ckpt_path=args.ckpt_path)
                model = GenHPFDownstreamModule(encoder=encoder,
                                            lr=learning_rate,
                                            max_epochs=75,
                                            num_outputs=1,
                                            pos_weight=1.0,
                                            prediction_csv_path=prediction_csv_path)
                
            elif args.backbone_name == 'remed':
                learning_rate = trial.suggest_float("learning_rate",  1e-4, 1e-2)
                weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)
                hparams={'learning_rate': learning_rate,
                        'weight_decay': weight_decay}
                train_dataset = REMedGenHPFPoolDataset(hf_path=args.data_path,
                                                    data_idx_path=args.data_idx_path,
                                                    seq_field=args.main_window,
                                                    time_field=args.time_field,
                                                    time_diff_field=args.time_diff_field,
                                                    label_field=args.task,
                                                    split='train',
                                                    tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                    seq_len=511,
                                                    max_tokens=64)
                val_dataset = REMedGenHPFPoolDataset(hf_path=args.data_path,
                                                    data_idx_path=args.data_idx_path,
                                                    seq_field=args.main_window,
                                                    time_field=args.time_field,
                                                    time_diff_field=args.time_diff_field,
                                                    label_field=args.task,
                                                    split='tuning',
                                                    tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                    seq_len=511,
                                                    max_tokens=64)
                collate_fn = REMedGenHPFCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
                encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                                        pad_token_id=train_dataset.tokenizer.pad_token_id,
                                        encoder_embed_dim=128,
                                        encoder_layers=2,
                                        encoder_ffn_embed_dim=512,
                                        encoder_attention_heads=4,
                                        agg_embed_dim=128,
                                        agg_layers=4,
                                        agg_ffn_embed_dim=512,
                                        agg_attention_heads=4,
                                        dropout=0.2,
                                        max_token_len=64,   
                                        max_events=511,
                                        encoder_only=True,
                                        ckpt_path=args.ckpt_path)
                remed_genhpf = REMedWithGenHPF(genhpf_encoder=encoder,
                                            pred_dim=512,
                                            num_classes=1,
                                            pred_time=args.pred_time,
                                            max_retrieve_len=128,
                                            n_heads=8,
                                            n_layers=2,
                                            dropout=0.2,
                                            freeze_encoder=True)
                model = REMedLightningModule(model=remed_genhpf,
                                            lr=learning_rate,
                                            wd=weight_decay,
                                            max_epochs=75,
                                            pos_weight=1.0,
                                            freeze_encoder=True,
                                            use_warmup=False,
                                            warmup_steps=500,
                                            num_classes=1,
                                            prediction_csv_path=prediction_csv_path)
            train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=24,
                                        shuffle=True,
                                        collate_fn=collate_fn,
                                        pin_memory=True,
                                        persistent_workers=True,
                                        pin_memory_device='cuda',
                                        prefetch_factor=4)
            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=24,
                                        shuffle=False,
                                        collate_fn=collate_fn,
                                        pin_memory=True,
                                        persistent_workers=True,
                                        pin_memory_device='cuda',
                                        prefetch_factor=4)


            # checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
            #                                         monitor='val_loss',
            #                                         mode='min',
            #                                         every_n_epochs=1,
            #                                         save_top_k=1)
            early_stop = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001,
                                    mode='min', 
                                    patience=5)

            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            torch.set_float32_matmul_precision('high')

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                
                precision = "bf16-mixed" if major >= 8 else "16-mixed"
            else:
                precision = '32-true'
            trainer = lt.Trainer(accelerator='auto', 
                                devices='auto',
                                strategy='auto',
                                logger=wandb_logger, 
                                log_every_n_steps=1,
                                num_sanity_val_steps=0,
                                max_epochs=75,
                                precision=precision,
                                callbacks=[early_stop,lr_monitor],#, checkpoint_callback],
                                enable_checkpointing=True
                                )
            trainer.logger.log_hyperparams(hparams)
            trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        except optuna.exceptions.TrialPruned:
            wandb.finish()
            raise
        finally:
            wandb.finish()

        return early_stop.best_score.item()

    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(study_name=args.backbone_name,
                                direction="minimize", 
                                storage=f'sqlite:////scratch/sas10092/ehr-foundation/models/optuna_dbs/{args.backbone_name}_{mode}_{args.task}.db',
                                pruner=pruner,
                                load_if_exists=True,
                                sampler=TPESampler())

    study.optimize(objective, n_trials=25,show_progress_bar=True,gc_after_trial=True)



elif args.run_mode == 'eval':
    learning_rate = 0.000269016592023212
    weight_decay = 0.0014355353210248
    dropout = 0.1

    if args.backbone_name == 'descemb':
        mode = 'cls-ft' if args.freeze else 'bert-ft'
    elif args.variant is not None:
        mode = args.variant
    else:
        mode = 'base'


    wandb.login(key=args.wandb_api_key)

    wandb_logger = WandbLogger(project=args.project_name,
                                save_dir=args.wandb_log_dir,
                                version=f"{args.backbone_name}_{mode}_{args.job_id}_{args.task}_{args.version}_{'final'}",
                                name=f"{args.backbone_name}_{mode}_{args.job_id}_{args.task}_{args.version}_{'final'}",
                                tags=[args.version,args.task,args.backbone_name,mode]) 

    run_dir = get_run_dir(wandb_logger)         
    ckpt_dir = os.path.join(run_dir, "ckpt")
    make_dir(ckpt_dir)
    prediction_csv_path = os.path.join(ckpt_dir,"test_predictions.csv")

    if args.backbone_name == 'bert' and args.variant == 'hibehrt':
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant='hibehrt')
        learning_rate = learning_rate
        hparams={'learning_rate': learning_rate}

        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                    chunk_length=args.seq_length,
                                    overlap=args.seq_overlap)

        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    limits_dict=limits,
                                    task=args.task,
                                    main_window=args.main_window,
                                    seq_length=args.seq_length + 1,
                                    use_numeric=args.use_numeric,
                                    add_cls=False,
                                    use_time=args.use_time,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train')
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    limits_dict=limits,
                                    task=args.task,
                                    main_window=args.main_window,
                                    seq_length=args.seq_length + 1,
                                    use_numeric=args.use_numeric,
                                    add_cls=False,
                                    use_time=args.use_time,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning')
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    limits_dict=limits,
                                    task=args.task,
                                    main_window=args.main_window,
                                    seq_length=args.seq_length + 1,
                                    use_numeric=args.use_numeric,
                                    add_cls=False,
                                    use_time=args.use_time,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out')
        collate_fn = HiBEHRTEvalCollator(seq_gen=seq_gen,
                                            chunk_length=256,
                                            overlap=32,
                                            add_cls_per_chunk=True)
        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        
        model = HiBEHRTModule(config=cfg,
                                backbone=ModelClass,
                                ckpt_path=args.ckpt_path,
                                lr=learning_rate,
                                max_epochs=75,
                                freeze=False,
                                pooling='mean',
                                dropout=0.1,
                                optimizer='sgd',
                                use_time=args.use_time,
                                use_numeric=args.use_numeric,
                                use_type=args.use_type,
                                use_visit=args.use_visit, 
                                use_stage=args.use_stage,
                                prediction_csv_path=prediction_csv_path)
        
    elif args.backbone_name in 'bert' and args.variant == 'medbert':
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
        learning_rate = learning_rate
        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                        chunk_length=args.seq_length,
                                        overlap=args.seq_overlap)
        hparams={'learning_rate': learning_rate}
        collate_fn = EvalCollator()
        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train')
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning')
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out')
        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        model = EvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path=args.ckpt_path,
                            lr=learning_rate,
                            max_epochs=75,
                            pooling='cls',
                            use_numeric=args.use_numeric,
                            use_time=args.use_time,
                            use_type=args.use_type,
                            use_visit=args.use_visit, 
                            use_stage=args.use_stage,
                            freeze=False,
                            optimizer='sgd',
                            prediction_csv_path=prediction_csv_path)
        
    elif args.backbone_name in 'bert' and args.variant == 'behrt':
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
        learning_rate = learning_rate
        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                        chunk_length=args.seq_length,
                                        overlap=args.seq_overlap)
        hparams={'learning_rate': learning_rate}
        collate_fn = EvalCollator()
        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train')
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning')
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out')
        
        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        model = EvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path=args.ckpt_path,
                            lr=learning_rate,
                            max_epochs=75,
                            pooling='cls',
                            use_numeric=args.use_numeric,
                            use_time=args.use_time,
                            use_type=args.use_type,
                            use_visit=args.use_visit, 
                            use_stage=args.use_stage,
                            freeze=False,
                            optimizer='sgd',
                            prediction_csv_path=prediction_csv_path)
        
    elif args.backbone_name in 'bert' and args.variant == 'cehrbert':
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=args.variant)
        learning_rate = learning_rate
        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                        chunk_length=args.seq_length,
                                        overlap=args.seq_overlap)
        hparams={'learning_rate': learning_rate}
        collate_fn = EvalCollator()
        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train')
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning')
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out')
        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        model = EvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path=args.ckpt_path,
                            lr=learning_rate,
                            max_epochs=75,
                            pooling='cls',
                            use_numeric=args.use_numeric,
                            use_time=args.use_time,
                            use_type=args.use_type,
                            use_visit=args.use_visit, 
                            use_stage=args.use_stage,
                            freeze=False,
                            optimizer='sgd',
                            prediction_csv_path=prediction_csv_path)   

    elif args.backbone_name == 'mamba':
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=None)
        learning_rate = learning_rate
        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                        chunk_length=args.seq_length,
                                        overlap=args.seq_overlap)
        hparams={'learning_rate': learning_rate}
        collate_fn = EvalCollator()
        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train')
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning')
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out')
        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        model = EvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path=args.ckpt_path,
                            lr=learning_rate,
                            wd=0.0,
                            max_epochs=75,
                            pooling='cls',
                            use_numeric=args.use_numeric,
                            use_time=args.use_time,
                            use_type=args.use_type,
                            use_visit=args.use_visit, 
                            use_stage=args.use_stage,
                            freeze=False,
                            optimizer='sgd',
                            prediction_csv_path=prediction_csv_path)
    elif args.backbone_name in ['roberta','longformer','big_bird','roformer','modernbert']:
        ConfigClass, ModelClass = get_config_and_model_cls(args.backbone_name, mode='eval', variant=None)
        learning_rate = learning_rate
        weight_decay = weight_decay
        seq_gen = SequencesGenerator(tokenizer_path=args.tokenizer_path,
                                        chunk_length=args.seq_length,
                                        overlap=args.seq_overlap)
        hparams={'learning_rate': learning_rate,
                'weight_decay': weight_decay}
        collate_fn = EvalCollator()
        train_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='train',
                                    use_long_context= True if args.seq_length > 1024 else False)
        val_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='tuning',
                                    use_long_context= True if args.seq_length >1024 else False)
        test_dataset = EvalDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    seq_gen=seq_gen,
                                    seq_length=args.seq_length,
                                    limits_dict=limits,
                                    main_window=args.main_window,
                                    task=args.task,
                                    use_time=args.use_time,
                                    use_numeric=args.use_numeric,
                                    use_type=args.use_type,
                                    use_visit=args.use_visit, 
                                    use_stage=args.use_stage,
                                    split='held_out',
                                    use_long_context= True if args.seq_length >1024 else False)

        cfg = ConfigClass(
            vocab_size=seq_gen.tokenizer.vocab_size,
            cls_token_id=seq_gen.tokenizer.cls_id,
            pad_token_id=seq_gen.tokenizer.pad_id,
            type_vocab_size=43,
            visit_vocab_size=990,
            stage_vocab_size=5,
            refernece_compile=False)
        cfg = fix_roberta_longformer_max_pos(cfg)
        model = EvalModel(config=cfg,
                            backbone=ModelClass,
                            ckpt_path=args.ckpt_path,
                            lr=learning_rate,
                            wd=weight_decay,
                            max_epochs=75,
                            pooling='cls',
                            use_numeric=args.use_numeric,
                            use_time=args.use_time,
                            use_type=args.use_type,
                            use_visit=args.use_visit, 
                            use_stage=args.use_stage,
                            freeze=False,
                            optimizer='sgd',
                            prediction_csv_path=prediction_csv_path)
        
    elif args.backbone_name == 'descemb':
        learning_rate = learning_rate
        dropout = dropout
        hparams={'learning_rate': learning_rate,
                    'dropout': dropout}
        train_dataset = DescEmbDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    task=args.task,
                                    main_window=args.main_window,
                                    max_word_len=12,
                                    max_events=510,
                                    split='train') 
        val_dataset = DescEmbDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    task=args.task,
                                    main_window=args.main_window,
                                    max_word_len=12,
                                    max_events=510,
                                    split='tuning') 
        test_dataset = DescEmbDataset(dataset_path=args.data_path,
                                    data_idx_path=args.data_idx_path,
                                    task=args.task,
                                    main_window=args.main_window,
                                    max_word_len=12,
                                    max_events=510,
                                    split='held_out') 
        collate_fn = DescEmbCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
        cfg = SimpleNamespace(bert_model_name="google/bert_uncased_L-2_H-128_A-2",
                                pred_embed_dim=128,
                                pred_hidden_dim=256,     
                                max_event_len=510,       
                                rnn_layer=1,
                                init_bert_random=False,  
                                task="binary")
        model = DescEmbEvalModel(config=cfg,
                                lr=learning_rate,
                                max_epochs=75,
                                dropout=dropout,
                                freeze=args.freeze,
                                prediction_csv_path=prediction_csv_path)
    elif args.backbone_name in ['genhpf']:
        learning_rate = learning_rate
        dropout = dropout
        hparams={'learning_rate': learning_rate,
                    'dropout': dropout}
        train_dataset = HierarchicalGenHPFDataset(
            dataset_path=args.data_path,
            data_idx_path=args.data_idx_path,
            seq_field=args.main_window,
            label_field=args.task,
            split='train',
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_events=510,
            max_tokens=64)
        val_dataset = HierarchicalGenHPFDataset(
            dataset_path=args.data_path,
            data_idx_path=args.data_idx_path,
            seq_field=args.main_window,
            label_field=args.task,
            split='tuning',
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_events=510,
            max_tokens=64)
        test_dataset = HierarchicalGenHPFDataset(
            dataset_path=args.data_path,
            data_idx_path=args.data_idx_path,
            seq_field=args.main_window,
            label_field=args.task,
            split='held_out',
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_events=510,
            max_tokens=64)
        collate_fn = GenHPFEvalCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
        encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                                pad_token_id=train_dataset.tokenizer.pad_token_id,
                                encoder_embed_dim=128,
                                encoder_layers=2,
                                encoder_ffn_embed_dim=512,
                                encoder_attention_heads=4,
                                agg_embed_dim=128,
                                agg_layers=4,
                                agg_ffn_embed_dim=512,
                                agg_attention_heads=4,
                                dropout=dropout,
                                max_token_len=64,   
                                max_events=510,
                                encoder_only=False,
                                ckpt_path=args.ckpt_path)
        model = GenHPFDownstreamModule(encoder=encoder,
                                        lr=learning_rate,
                                        max_epochs=75,
                                        num_outputs=1,
                                        pos_weight=1.0,
                                        prediction_csv_path=prediction_csv_path)
        
    elif args.backbone_name == 'remed':
        learning_rate = learning_rate
        weight_decay = weight_decay
        hparams={'learning_rate': learning_rate,
                    'weight_decay': weight_decay}
        train_dataset = REMedGenHPFPoolDataset(hf_path=args.data_path,
                                                data_idx_path=args.data_idx_path,
                                                seq_field=args.main_window,
                                                time_field=args.time_field,
                                                time_diff_field=args.time_diff_field,
                                                label_field=args.task,
                                                split='train',
                                                tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                seq_len=510,
                                                max_tokens=64)
        val_dataset = REMedGenHPFPoolDataset(hf_path=args.data_path,
                                                data_idx_path=args.data_idx_path,
                                                seq_field=args.main_window,
                                                time_field=args.time_field,
                                                time_diff_field=args.time_diff_field,
                                                label_field=args.task,
                                                split='tuning',
                                                tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                seq_len=510,
                                                max_tokens=64)
        test_dataset = REMedGenHPFPoolDataset(hf_path=args.data_path,
                                                data_idx_path=args.data_idx_path,
                                                seq_field=args.main_window,
                                                time_field=args.time_field,
                                                time_diff_field=args.time_diff_field,
                                                label_field=args.task,
                                                split='held_out',
                                                tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                                seq_len=510,
                                                max_tokens=64)
        collate_fn = REMedGenHPFCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
        encoder = GenHPFEncoder(vocab_size=train_dataset.tokenizer.vocab_size,
                                pad_token_id=train_dataset.tokenizer.pad_token_id,
                                encoder_embed_dim=128,
                                encoder_layers=2,
                                encoder_ffn_embed_dim=512,
                                encoder_attention_heads=4,
                                agg_embed_dim=128,
                                agg_layers=4,
                                agg_ffn_embed_dim=512,
                                agg_attention_heads=4,
                                dropout=0.2,
                                max_token_len=64,   
                                max_events=510,
                                encoder_only=True,
                                ckpt_path=args.ckpt_path)
        remed_genhpf = REMedWithGenHPF(genhpf_encoder=encoder,
                                        pred_dim=512,
                                        num_classes=1,
                                        pred_time=args.pred_time,
                                        max_retrieve_len=128,
                                        n_heads=8,
                                        n_layers=2,
                                        dropout=0.2,
                                        freeze_encoder=True)
        model = REMedLightningModule(model=remed_genhpf,
                                        lr=learning_rate,
                                        wd=weight_decay,
                                        max_epochs=75,
                                        pos_weight=1.0,
                                        freeze_encoder=True,
                                        use_warmup=False,
                                        warmup_steps=500,
                                        num_classes=1,
                                        prediction_csv_path=prediction_csv_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                num_workers=8,
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=8,
                                shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                num_workers=8,
                                shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)


    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                            monitor='val_loss',
                                            mode='min',
                                            every_n_epochs=1,
                                            save_top_k=1)
    early_stop = EarlyStopping(monitor='val_loss',
                                min_delta=0.001,
                                mode='min', 
                                patience=3)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    torch.set_float32_matmul_precision('high')

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        
        precision = "bf16-mixed" if major >= 8 else "16-mixed"
    else:
        precision = '32-true'
    trainer = lt.Trainer(accelerator='auto', 
                        devices='auto',
                        strategy='auto',
                        logger=wandb_logger, 
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        max_epochs=75,
                        precision=precision,
                        callbacks=[early_stop,lr_monitor, checkpoint_callback],
                        enable_checkpointing=True
                        )
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')