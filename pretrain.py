import os
import torch
import wandb
import argparse
import functools
import lightning.pytorch as lt

from torch.utils.data import DataLoader
from lightning.pytorch.tuner import Tuner
from src.models.models import MLMPretraining
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from src.models.utils import get_config_and_model_cls, fix_roberta_longformer_max_pos
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.models.baseline_models import EHRMambaNTPPretraining, GenHPFEncoder, GenHPFSimCLRModule
from src.data.datasets import SequencesGenerator, EHRPretrainDataset, MLMDataCollator, PROTECTED_TOKENS
from src.data.baseline_datasets import CausalLMDataCollator, GenHPFSimCLRCollator, GenHPFSimCLRDataset

parser = argparse.ArgumentParser(description='pretraining command line interface')

parser.add_argument('--learning-rate', type=float, default=0.00001)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--chunk-length', type=int, default=1536,)
parser.add_argument('--overlap', type=int, default=192,)

args = parser.parse_args()
tokenizer_path = os.getenv('TOKENIZER_PATH')
data_path = os.getenv('DATA_PATH')
data_idx_path = os.getenv('DATA_IDX_PATH')
wandb_api_key = os.getenv('WANDB_API_KEY')
backbone_name = os.getenv('BACKBONE')
log_dir = os.getenv('LOG_DIR')
version = os.getenv('VERSION')
job_id = os.getenv('SLURM_JOB_ID')
pretrain_mode = os.getenv('PRETRAIN_MODE')
baseline = os.getenv('BASELINE')

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
    
if pretrain_mode in ["mlm","causal"]:
    ConfigClass, ModelClass = get_config_and_model_cls(backbone_name,mode=pretrain_mode,variant=None if baseline in ['transformer','genhpf'] else baseline)
    seq_gen = SequencesGenerator(tokenizer_path= tokenizer_path,
                                chunk_length=args.chunk_length,
                                overlap=args.overlap,)
    train_dataset = EHRPretrainDataset(dataset_path=data_path,
                                       data_idx_path=data_idx_path,
                                       seq_generator=seq_gen,
                                       split='train')
    val_dataset = EHRPretrainDataset(dataset_path=data_path,
                                    data_idx_path=data_idx_path,
                                    seq_generator=seq_gen,
                                    split='tuning')
    if pretrain_mode == "mlm":
        collate_fn_train = MLMDataCollator(tokenizer=seq_gen.tokenizer,
                                           protected_tokens=PROTECTED_TOKENS,
                                           mask_prob=0.15,
                                           replace_prob=0.8,
                                           random_prob=0.1,
                                           structural_dropout=0.15)
        collate_fn_tune = MLMDataCollator(tokenizer=seq_gen.tokenizer,
                                          protected_tokens=PROTECTED_TOKENS,
                                          mask_prob=0.15,
                                          replace_prob=0.8,
                                          random_prob=0.1,
                                          structural_dropout=0.0)
        train_dataoader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn_train,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)

        val_dataoader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn_tune,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)
    elif pretrain_mode == "causal":
        collate_fn = CausalLMDataCollator(tokenizer=seq_gen.tokenizer)
        train_dataoader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)

        val_dataoader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=4)
    if pretrain_mode == "mlm":
        cfg = ConfigClass(vocab_size=seq_gen.tokenizer.vocab_size,
                          cls_token_id=seq_gen.tokenizer.cls_id,
                          pad_token_id=seq_gen.tokenizer.pad_id,
                          type_vocab_size=43,
                          visit_vocab_size=990,
                          stage_vocab_size=5,
                          refernece_compile=False)
    elif pretrain_mode == "causal":
        cfg = ConfigClass(vocab_size=seq_gen.tokenizer.vocab_size,
                          pad_token_id=seq_gen.tokenizer.pad_id,
                          type_vocab_size=43,
                          visit_vocab_size=990,
                          stage_vocab_size=5,
                          use_mambapy=True)
    cfg = fix_roberta_longformer_max_pos(cfg)

    if pretrain_mode == "causal":
        model = EHRMambaNTPPretraining(config=cfg,
                                       backbone=ModelClass,
                                       lr=args.learning_rate,
                                       wd=args.weight_decay,
                                       max_epochs=args.max_epochs,
                                       use_type=True,
                                       use_visit=True,
                                       use_stage=True,
                                       use_time=True,
                                       use_numeric=True)
    elif pretrain_mode == "mlm":    
        model = MLMPretraining(config=cfg,
                               backbone=ModelClass,
                               lr=args.learning_rate,
                               wd=args.weight_decay,
                               max_epochs=args.max_epochs,
                               use_type=True,
                               use_visit=True,
                               use_stage=True,
                               use_time=True,
                               use_numeric=True)
        
elif pretrain_mode == "simclr":
    train_dataset= GenHPFSimCLRDataset(dataset_path=data_path,
                                       data_idx_path=data_idx_path,
                                       seq_field='within_stay_remed',
                                       split="train",
                                       tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                       max_events=510,
                                       max_tokens=64)
    val_dataset= GenHPFSimCLRDataset(dataset_path=data_path,
                                     data_idx_path=data_idx_path,
                                     seq_field='within_stay_remed',
                                     split="tuning",
                                     tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                                     max_events=510,
                                     max_tokens=64)
    
    collate_fn = GenHPFSimCLRCollator(pad_token_id=train_dataset.tokenizer.pad_token_id,
                                      mask_token_id=train_dataset.tokenizer.mask_token_id,
                                      cls_token_id=train_dataset.tokenizer.cls_token_id,
                                      mask_prob=0.15)
    train_dataoader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,          
                                 shuffle=True,
                                 num_workers=8,
                                 collate_fn=collate_fn,
                                 persistent_workers=True,
                                 prefetch_factor=4)
    
    val_dataoader = DataLoader(val_dataset,
                                 batch_size=args.batch_size,          
                                 shuffle=True,
                                 num_workers=8,
                                 collate_fn=collate_fn,
                                 persistent_workers=True,
                                 prefetch_factor=4)
    
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
                            dropout=0.3,
                            max_token_len=64,   
                            max_events=510,
                            encoder_only=False)

    model = GenHPFSimCLRModule(encoder=encoder,
                               lr=args.learning_rate,
                               wd=args.weight_decay,
                               max_epochs=args.max_epochs,
                               temperature=0.1)
    
wandb.login(key=wandb_api_key)
wandb_logger = WandbLogger(project='ehr-ragp-pretraining',
                           save_dir=log_dir,
                           version=f'{backbone_name}_{baseline}_{job_id}_{args.chunk_length}_{args.overlap}_{version}',
                           name=f'{backbone_name}_{baseline}_{job_id}_{args.learning_rate}_{args.chunk_length}_{args.overlap}_{version}',
                           tags=[backbone_name,version])
run_dir = get_run_dir(wandb_logger)         
ckpt_dir = os.path.join(run_dir, "ckpt")
make_dir(ckpt_dir)
checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                      monitor='val_loss', 
                                      mode='min',
                                      every_n_epochs=1,
                                      save_top_k=3,)
early_stop = EarlyStopping(monitor='val_loss', 
                        min_delta=0.0005,
                        mode='min', 
                        patience=5)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
def main():
    torch.set_float32_matmul_precision('high')
    trainer = lt.Trainer(accelerator='gpu', 
                        devices='auto',
                        strategy='ddp_find_unused_parameters_true',
                        logger=wandb_logger, 
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        max_epochs=args.max_epochs,
                        enable_progress_bar=True,
                        precision='16-mixed', 
                        callbacks=[checkpoint_callback,early_stop,lr_monitor]
                        )


    trainer.fit(model=model, train_dataloaders=train_dataoader, val_dataloaders=val_dataoader)

if __name__ == "__main__":
    main()









