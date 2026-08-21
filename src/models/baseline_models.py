
import math
import copy
import torch
import pandas as pd
import torch.nn as nn
import lightning.pytorch as lt
import torch.nn.functional as F

from .models import EHREmbeddings
from torchmetrics import Accuracy
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from .utils import log_bootstrap_ci_text_percentile
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers.models.roformer.modeling_roformer import RoFormerConfig, RoFormerEncoder


#########################################################
# DescEmb model
#########################################################

class BertEventEncoder(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        pred_embed_dim: int = 128,
        init_bert_random: bool = False,
    ):
        super().__init__()

        if init_bert_random:
            config = AutoConfig.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = AutoModel.from_pretrained(bert_model_name)

        hidden_size = self.bert.config.hidden_size
        self.post_encode_proj = nn.Linear(hidden_size, pred_embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, S, W = input_ids.shape

        flat_ids = input_ids.view(B * S, W)
        flat_mask = attention_mask.view(B * S, W)

        outputs = self.bert(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :] 

        event_emb = self.post_encode_proj(cls_emb)  
        event_emb = event_emb.view(B, S, -1)        
        return event_emb
    

class GRUEventHead(nn.Module):

    def __init__(self, pred_embed_dim: int=128, 
                 pred_hidden_dim: int=256, 
                 max_event_len: int =511,
                 n_layers: int = 1, 
                 dropout: float = 0.3, 
                 task: str = "binary"):
        super().__init__()
        self.pred_embed_dim = pred_embed_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.n_layers = n_layers
        self.max_event_len = max_event_len
        self.task = task

        self.model = nn.GRU(
            input_size=self.pred_embed_dim,
            hidden_size=self.pred_hidden_dim,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
            num_layers=self.n_layers,
        )

        out_dim = 18 if task == "diagnosis" else 1
        self.final_proj = nn.Linear(self.pred_hidden_dim, out_dim)

    def pack_pad_seq(self, x: torch.Tensor, lengths: torch.Tensor):
        lengths = lengths.view(-1).cpu()
        lengths[lengths > self.max_event_len] = self.max_event_len

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.model(packed)
        output_seq, output_len = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0
        )
        return output_seq, output_len

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
       
        self.model.flatten_parameters()

        output_seq, _ = self.pack_pad_seq(x, seq_len) 
        i = range(x.size(0))
        last_hidden = output_seq[i, -1, :]             

        logits = self.final_proj(last_hidden)          
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)                
        return logits
    

class DescEmbEvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        lr: float = 2e-5,
        wd: float = 0.0,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        prediction_csv_path: str = '.'
    ):
        super().__init__()
        self.save_hyperparameters()

        bert_model_name = getattr(config, "bert_model_name", "google/bert_uncased_L-2_H-128_A-2")
        pred_embed_dim = getattr(config, "pred_embed_dim", 128)
        pred_hidden_dim = getattr(config, "pred_hidden_dim", 256)
        max_event_len = getattr(config, "max_event_len", 511)
        task = getattr(config, "task", "binary")

        self.encoder = BertEventEncoder(
            bert_model_name=bert_model_name,
            pred_embed_dim=pred_embed_dim,
            init_bert_random=getattr(config, "init_bert_random", False),
        )
        self.classifier = GRUEventHead(
            pred_embed_dim=pred_embed_dim,
            pred_hidden_dim=pred_hidden_dim,
            max_event_len=max_event_len,
            n_layers=getattr(config, "rnn_layer", 1),
            dropout=dropout,
            task=task,
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.classifier.parameters():
                p.requires_grad = True

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.criterion = nn.BCEWithLogitsLoss()
        self.prediction_csv_path = prediction_csv_path
        self.train_step_preds = []
        self.train_step_labels = []
        self.val_step_preds = []
        self.val_step_labels = []
        self.test_step_preds = []
        self.test_step_logits = [] 
        self.test_step_labels = []
        self.test_step_subject_ids = []
        self.test_step_hadm_ids = []
        self.test_step_icustay_ids = []

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

    def forward(self, input_ids, attention_mask, seq_len=None, labels=None):
        event_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  

        if seq_len is None:
            event_mask = attention_mask.any(dim=-1)     
            seq_len = event_mask.sum(dim=-1)             
        logits = self.classifier(event_emb, seq_len)       
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)

        pos_score = torch.sigmoid(logits)

        self.train_step_labels.append(y.detach())
        self.train_step_preds.append(pos_score.detach())

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_labels) == 0:
            return

        y = torch.cat(self.train_step_labels)
        pos_score = torch.cat(self.train_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.train_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.train_auprc(gathered_pos_score, gathered_y.long())

        self.log("train_auroc", auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("train_auprc", auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.train_step_labels.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.val_step_labels.append(y.detach())
        self.val_step_preds.append(pos_score.detach())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        if len(self.val_step_labels) == 0:
            return

        y = torch.cat(self.val_step_labels)
        pos_score = torch.cat(self.val_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.val_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.val_auprc(gathered_pos_score, gathered_y.long())

        self.log("val_auroc", auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auprc", auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.val_step_labels.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            seq_len=batch["seq_len"],
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.test_step_logits.append(logits.detach())
        self.test_step_preds.append(pos_score.detach())
        self.test_step_labels.append(y.detach())

        self.test_step_subject_ids.append(batch["subject_id"])
        self.test_step_hadm_ids.append(batch["hadm_id"])
        self.test_step_icustay_ids.append(batch["icustay_id"])

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss


    def on_test_epoch_end(self,*arg, **kwargs) -> None:
        if len(self.test_step_labels) == 0:
            return
        y = torch.cat(self.test_step_labels)
        pos_score = torch.cat(self.test_step_preds)
        logits = torch.cat(self.test_step_logits)
        subject_ids = torch.cat(self.test_step_subject_ids)
        hadm_ids = torch.cat(self.test_step_hadm_ids)
        icustay_ids = torch.cat(self.test_step_icustay_ids)

        gathered_y = self.all_gather(y)
        gathered_pos_score = self.all_gather(pos_score)
        gathered_logits = self.all_gather(logits)

        gathered_subject_ids = self.all_gather(subject_ids)
        gathered_hadm_ids = self.all_gather(hadm_ids)
        gathered_icustay_ids = self.all_gather(icustay_ids)

        gathered_y = gathered_y.reshape(-1)
        gathered_pos_score = gathered_pos_score.reshape(-1)
        gathered_logits = gathered_logits.reshape(-1)

        gathered_subject_ids = gathered_subject_ids.reshape(-1)
        gathered_hadm_ids = gathered_hadm_ids.reshape(-1)
        gathered_icustay_ids = gathered_icustay_ids.reshape(-1)

        auroc = self.test_auroc(gathered_pos_score,gathered_y.long())
        auprc = self.test_auprc(gathered_pos_score,gathered_y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True,)
        self.log("test_auprc", auprc, on_epoch=True,logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=gathered_y,
            y_score=gathered_pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3)

        if self.global_rank == 0:

            results = pd.DataFrame(
                {
                    "subject_id": gathered_subject_ids.cpu().numpy(),
                    "hadm_id": gathered_hadm_ids.cpu().numpy(),
                    "icustay_id": gathered_icustay_ids.cpu().numpy(),
                    "label": gathered_y.cpu().numpy(),
                    "prediction": gathered_pos_score.cpu().numpy(),
                    "logit": gathered_logits.cpu().numpy(),
                }
            )

            results.to_csv(self.prediction_csv_path, index=False)

        self.test_step_labels.clear()
        self.test_step_preds.clear()
        self.test_step_logits.clear()
        self.test_step_subject_ids.clear()
        self.test_step_hadm_ids.clear()
        self.test_step_icustay_ids.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     eta_min=0,
        #     T_max=self.max_epochs,
        # )
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}
    

#########################################################
# GenHPF model
#########################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                      
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)                              
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
class GenHPFEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        encoder_embed_dim: int = 128,
        encoder_layers: int = 2,
        encoder_ffn_embed_dim: int = 512,
        encoder_attention_heads: int = 4,
        agg_embed_dim: int = 128,
        agg_layers: int = 4,
        agg_ffn_embed_dim: int = 512,
        agg_attention_heads: int = 4,
        dropout: float = 0.1,
        max_token_len: int = 128,
        max_events: int = 511,
        encoder_only: bool = False,
        ckpt_path: str = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_only = encoder_only
        self.pad_token_id = pad_token_id
        self.encoder_embed_dim = encoder_embed_dim
        self.agg_embed_dim = agg_embed_dim

        self.word_embeddings = nn.Embedding(vocab_size, 
                                            encoder_embed_dim, 
                                            padding_idx=pad_token_id)

        
        self.token_pos_encoding = PositionalEncoding(d_model=encoder_embed_dim, 
                                                     dropout=dropout, 
                                                     max_len=max_token_len)

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embed_dim,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers
        )

        
        self.post_encode_proj = nn.Linear(encoder_embed_dim, agg_embed_dim)

        
        self.event_pos_encoding = PositionalEncoding(
            d_model=agg_embed_dim, dropout=dropout, max_len=max_events
        )

        agg_layer = nn.TransformerEncoderLayer(
            d_model=agg_embed_dim,
            nhead=agg_attention_heads,
            dim_feedforward=agg_ffn_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.event_aggregator = nn.TransformerEncoder(
            agg_layer, num_layers=agg_layers
        )

        self.event_layer_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-12)
        self.agg_layer_norm = nn.LayerNorm(agg_embed_dim, eps=1e-12)
        
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("model.encoder."):
                    cleaned[k.replace("model.encoder.", "")] = v
                elif k.startswith("encoder."):
                    cleaned[k.replace("encoder.", "")] = v
                else:
                    cleaned[k] = v

            missing, unexpected = self.load_state_dict(cleaned, strict=False)
            print(f"missing= \n{missing}")
            print(f"unexpected= \n{unexpected}")

    def forward(
        self,
        input_ids: torch.Tensor,         
        padding_mask: Optional[torch.Tensor] = None,  
#         encoder_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, W = input_ids.shape

        flat_ids = input_ids.view(B * S, W)


        x_tok = self.word_embeddings(flat_ids)
        x_tok = self.token_pos_encoding(x_tok)
        x_tok = self.event_layer_norm(x_tok)

        token_pad_mask = flat_ids.eq(self.pad_token_id) 


        x_tok = self.event_encoder(
            x_tok, src_key_padding_mask=token_pad_mask
        ) 

        if token_pad_mask.any():
            x_tok = x_tok.masked_fill(token_pad_mask.unsqueeze(-1), 0.0)
            lengths = (~token_pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        else:
            lengths = torch.full(
                (B * S, 1), W, device=x_tok.device, dtype=torch.long
            )

        event_emb = x_tok.sum(dim=1) / lengths 

    
        event_emb = self.post_encode_proj(event_emb) 
        event_emb = event_emb.view(B, S, -1)        

        if padding_mask is None:
            padding_mask = input_ids.eq(self.pad_token_id).all(dim=2)  


        event_emb = self.event_pos_encoding(event_emb)
        event_emb = self.agg_layer_norm(event_emb)

        if self.encoder_only:
            return event_emb, padding_mask

        x = self.event_aggregator(
            event_emb, src_key_padding_mask=padding_mask
        ) 

        return x, padding_mask
    

class GenHPFSimCLRModel(nn.Module):
    def __init__(self, encoder: GenHPFEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder

        D = encoder.agg_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, proj_dim),
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)

        if pad_mask is not None and pad_mask.any():
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths

        z = self.proj(pooled)
        return z
    


class GenHPFClassifier(nn.Module):

    def __init__(
        self,
        encoder: GenHPFEncoder,
        num_outputs: int = 1,  
    ):
        super().__init__()
        self.encoder = encoder
        self.num_outputs = num_outputs

        self.classifier = nn.Linear(encoder.agg_embed_dim, num_outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        x, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)


        if pad_mask is not None and pad_mask.any():
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


        lengths = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths

        logits = self.classifier(pooled)  
        if self.num_outputs == 1:
            logits = logits.squeeze(-1) 
        return logits
    

class GenHPFDownstreamModule(lt.LightningModule):
    def __init__(
        self,
        encoder: GenHPFEncoder,
        num_outputs: int = 1,
        lr: float = 2e-5,
        wd: float = 1e-3,
        max_epochs: int = 100,
        pos_weight: float = 1.0,
        prediction_csv_path: str = '.'  
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.model = GenHPFClassifier(
            encoder=encoder,
            num_outputs=num_outputs,
        )

        # loss
        if num_outputs == 1:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        if num_outputs == 1:
            self.train_auroc = BinaryAUROC()
            self.train_auprc = BinaryAveragePrecision()
            self.val_auroc = BinaryAUROC()
            self.val_auprc = BinaryAveragePrecision()
            self.test_auroc = BinaryAUROC()
            self.test_auprc = BinaryAveragePrecision()

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.prediction_csv_path = prediction_csv_path
        self.train_step_preds = []
        self.train_step_labels = []
        self.val_step_preds = []
        self.val_step_labels = []
        self.test_step_preds = []
        self.test_step_logits = [] 
        self.test_step_labels = []
        self.test_step_subject_ids = []
        self.test_step_hadm_ids = []
        self.test_step_icustay_ids = []

    def forward(self, input_ids, padding_mask):
        logits = self.model(input_ids=input_ids, padding_mask=padding_mask)
        return logits


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]       
        padding_mask = batch["padding_mask"] 
        y = batch["label"].float().view(-1) 

        logits = self.forward(input_ids, padding_mask) 

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            self.train_step_labels.append(y.detach())
            self.train_step_preds.append(pos_score.detach())
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return

        y = torch.cat(self.train_step_labels)
        pos_score = torch.cat(self.train_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.train_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.train_auprc(gathered_pos_score, gathered_y.long())

        self.log("train_auroc", auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("train_auprc", auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.train_step_labels.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        y = batch["label"].float().view(-1)

        logits = self.forward(input_ids, padding_mask)

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            self.val_step_labels.append(y.detach())
            self.val_step_preds.append(pos_score.detach())
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return

        y = torch.cat(self.val_step_labels)
        pos_score = torch.cat(self.val_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.val_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.val_auprc(gathered_pos_score, gathered_y.long())

        self.log("val_auroc", auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auprc", auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.val_step_labels.clear()
        self.val_step_preds.clear()


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        y = batch["label"].float().view(-1)

        logits = self.forward(input_ids, padding_mask)

        if self.hparams.num_outputs == 1:
            loss = self.criterion(logits, y)
            pos_score = torch.sigmoid(logits)
            
            self.test_step_logits.append(logits.detach())
            self.test_step_preds.append(pos_score.detach())
            self.test_step_labels.append(y.detach())

            self.test_step_subject_ids.append(batch["subject_id"])
            self.test_step_hadm_ids.append(batch["hadm_id"])
            self.test_step_icustay_ids.append(batch["icustay_id"])
        else:
            y_long = y.long()
            loss = self.criterion(logits, y_long)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        if self.hparams.num_outputs != 1:
            return
        y = torch.cat(self.test_step_labels)
        pos_score = torch.cat(self.test_step_preds)
        logits = torch.cat(self.test_step_logits)
        subject_ids = torch.cat(self.test_step_subject_ids)
        hadm_ids = torch.cat(self.test_step_hadm_ids)
        icustay_ids = torch.cat(self.test_step_icustay_ids)

        gathered_y = self.all_gather(y)
        gathered_pos_score = self.all_gather(pos_score)
        gathered_logits = self.all_gather(logits)

        gathered_subject_ids = self.all_gather(subject_ids)
        gathered_hadm_ids = self.all_gather(hadm_ids)
        gathered_icustay_ids = self.all_gather(icustay_ids)

        gathered_y = gathered_y.reshape(-1)
        gathered_pos_score = gathered_pos_score.reshape(-1)
        gathered_logits = gathered_logits.reshape(-1)

        gathered_subject_ids = gathered_subject_ids.reshape(-1)
        gathered_hadm_ids = gathered_hadm_ids.reshape(-1)
        gathered_icustay_ids = gathered_icustay_ids.reshape(-1)

        auroc = self.test_auroc(gathered_pos_score,gathered_y.long())
        auprc = self.test_auprc(gathered_pos_score,gathered_y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True,)
        self.log("test_auprc", auprc, on_epoch=True,logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=gathered_y,
            y_score=gathered_pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3)

        if self.global_rank == 0:

            results = pd.DataFrame(
                {
                    "subject_id": gathered_subject_ids.cpu().numpy(),
                    "hadm_id": gathered_hadm_ids.cpu().numpy(),
                    "icustay_id": gathered_icustay_ids.cpu().numpy(),
                    "label": gathered_y.cpu().numpy(),
                    "prediction": gathered_pos_score.cpu().numpy(),
                    "logit": gathered_logits.cpu().numpy(),
                }
            )

            results.to_csv(self.prediction_csv_path, index=False)

        self.test_step_labels.clear()
        self.test_step_preds.clear()
        self.test_step_logits.clear()
        self.test_step_subject_ids.clear()
        self.test_step_hadm_ids.clear()
        self.test_step_icustay_ids.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = CosineAnnealingLR(
#             optimizer=optimizer,
#             eta_min=0.0,
#             T_max=self.max_epochs,
#         )
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}
    

class GenHPFSimCLRModule(lt.LightningModule):
    def __init__(
        self,
        encoder: GenHPFEncoder,
        lr: float = 1e-4,
        wd: float = 0.0,
        max_epochs: int = 100,
        temperature: float = 0.1,
        log_sim_every_n_steps: int = 20,   # <-- add
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.model = GenHPFSimCLRModel(encoder=encoder)
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.log_sim_every_n_steps = log_sim_every_n_steps  # <-- add

    def forward(self, input_ids, padding_mask):
        return self.model(input_ids=input_ids, padding_mask=padding_mask)

    def _nt_xent_loss(self, z: torch.Tensor) -> torch.Tensor:
        B2, D = z.shape
        assert B2 % 2 == 0
        B = B2 // 2

        z1 = F.normalize(z[:B], dim=1)
        z2 = F.normalize(z[B:], dim=1)
        reps = torch.cat([z1, z2], dim=0)                     

        sim = (reps @ reps.T) / self.temperature              
        diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(diag, float("-inf"))

        labels = torch.arange(2 * B, device=sim.device)
        labels = (labels + B) % (2 * B)

        return F.cross_entropy(sim, labels)

    @torch.no_grad()
    def log_simclr_cosines(self, z: torch.Tensor, prefix="simclr"):

        B2 = z.size(0)
        assert B2 % 2 == 0
        B = B2 // 2

        z = F.normalize(z, dim=1)
        z1, z2 = z[:B], z[B:]

        
        pos = F.cosine_similarity(z1, z2, dim=1) 
        pos_mean = pos.mean()

       
        reps = torch.cat([z1, z2], dim=0)                
        sim = reps @ reps.T                               

        diag = torch.eye(2*B, device=z.device, dtype=torch.bool)
        pos_mask = diag.roll(shifts=B, dims=1)            
        neg_mask = ~(diag | pos_mask)

        neg_mean = sim[neg_mask].mean()

        self.log(f"{prefix}_pos_cos", pos_mean, prog_bar=True, on_step=True, logger=True)
        self.log(f"{prefix}_neg_cos", neg_mean, prog_bar=True, on_step=True, logger=True)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]

        z = self.forward(input_ids, padding_mask)
        self.log_simclr_cosines(z)
        with torch.no_grad():
            z0 = F.normalize(z, dim=1)
            sim = z0 @ z0.T                     
            sim.fill_diagonal_(float("nan"))    
            mask = ~torch.isnan(sim)
            sim_vals = sim[mask]                 
            self.log("sim_mean", sim_vals.mean(), on_step=True, prog_bar=True)
            self.log("sim_std",  sim_vals.std(unbiased=False), on_step=True, prog_bar=True)
        loss = self._nt_xent_loss(z)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        padding_mask = batch["padding_mask"]
        z = self.forward(input_ids, padding_mask)
        self.log_simclr_cosines(z)
        with torch.no_grad():
            z0 = F.normalize(z, dim=1)
            sim = z0 @ z0.T                     
            sim.fill_diagonal_(float("nan"))    
            mask = ~torch.isnan(sim)
            sim_vals = sim[mask]                 
            self.log("val_sim_mean", sim_vals.mean(), on_step=True, prog_bar=True)
            self.log("val_sim_std",  sim_vals.std(unbiased=False), on_step=True, prog_bar=True)
        loss = self._nt_xent_loss(z)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.lr)}
    
#########################################################
# REMed model
#########################################################


class Retriever(nn.Module):
    def __init__(self, pred_dim: int):
        super().__init__()
        in_dim = pred_dim + 1  # repr + time
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim // 8),
            nn.LayerNorm(in_dim // 8),
            nn.ReLU(),
            nn.Linear(in_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        times = times.unsqueeze(-1).type(reprs.dtype)
        return self.model(torch.cat([reprs, times], dim=-1)).squeeze(-1)  # (B, L)


class ReprTimeEnc(nn.Module):
    def __init__(self, pred_dim: int, dropout: float, pred_time: int):
        super().__init__()
        self.pred_time = pred_time
        div_term = torch.exp(torch.arange(0, pred_dim, 2) * (-math.log(10000.0) / pred_dim))
        self.register_buffer("div_term", div_term)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, times: torch.Tensor):

        times = self.pred_time * 60 - times
        src_pad_mask = x.eq(0).all(dim=-1)  # (B, L)

        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * self.div_term)
        x = self.dropout(x + pe)
        return x, src_pad_mask


class Predictor(nn.Module):
    def __init__(self, 
                 pred_dim: int=512, 
                 dropout: float=0.2, 
                 pred_time: int=48, 
                 n_layers: int=2, 
                 n_heads: int=8, 
                 max_len: int=128):
        super().__init__()
        self.time_enc = ReprTimeEnc(pred_dim, dropout, pred_time)
        config = RoFormerConfig(
            hidden_size=pred_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=pred_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_len,
        )
        self.model = RoFormerEncoder(config)

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        x, src_pad_mask = self.time_enc(reprs, times)

        mask = src_pad_mask * torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
        mask = mask.unsqueeze(-1).unsqueeze(1)  
        return self.model(x, attention_mask=mask)["last_hidden_state"] 


class PredOutPutLayer(nn.Module):
    def __init__(self, pred_dim: int, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.final_proj = nn.Linear(pred_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0).any(dim=-1).unsqueeze(-1)  
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.final_proj(pooled)  
        return logits



class REMed(nn.Module):
    def __init__(
        self,
        pred_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        max_retrieve_len: int = 128,
        pred_time: int = 48,
        num_classes: int = 1,
    ):
        super().__init__()
        self.pred_dim = pred_dim
        self.max_retrieve_len = max_retrieve_len

        self.predictor = Predictor(pred_dim, dropout, pred_time, n_layers, n_heads, max_retrieve_len)
        self.emb2out_model = PredOutPutLayer(pred_dim, num_classes=num_classes)
        self.retriever = Retriever(pred_dim)

        self.register_buffer("random_token_emb", torch.randn(pred_dim))
        self.set_mode("retriever")

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == "retriever":
            self.requires_grad_(False)
            self.retriever.requires_grad_(True)
        elif mode == "predictor":
            self.requires_grad_(True)
            self.retriever.requires_grad_(False)

    def forward(self, reprs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:

        reprs = nn.functional.pad(reprs, (0, 0, 0, self.max_retrieve_len))
        times = nn.functional.pad(times, (0, self.max_retrieve_len))


        times = torch.where(reprs.eq(0).all(dim=-1), torch.tensor(1e10, device=times.device, dtype=times.dtype), times)

        sim = self.retriever(reprs, times)  # (B, L+Kpad)

        _sim = torch.where(reprs.eq(0).all(dim=-1), torch.zeros_like(sim), sim)
        topk_values, topk_indices = torch.topk(_sim, self.max_retrieve_len, dim=1)

        topk = torch.gather(reprs, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.pred_dim))
        topk_times = torch.gather(times, 1, topk_indices)
        B, K, E = topk.shape


        topk_times, order = topk_times.sort(dim=1)
        topk = topk.gather(1, order.unsqueeze(-1).repeat(1, 1, E))
        topk_values = topk_values.gather(1, order)

        def _retriever_path():
            _topk_values = topk_values.reshape(B * K, 1)
            _topk = topk.reshape(B * K, 1, -1)
            _topk_times = topk_times.reshape(B * K, 1)

            zero_idcs = _topk.eq(0).all(dim=-1)
            _topk_times = torch.where(zero_idcs, torch.zeros_like(_topk_times), _topk_times)
            _topk_values = torch.where(zero_idcs, torch.zeros_like(_topk_values), _topk_values)


            _topk = torch.where(zero_idcs.unsqueeze(-1), self.random_token_emb.expand(B * K, 1, E), _topk)
            _topk_values = _topk_values + 1e-10
            _topk_values = (_topk_values.reshape(B, K) / _topk_values.reshape(B, K).sum(dim=1, keepdim=True)).reshape(B * K)

            enc = self.predictor(_topk, _topk_times)
            logits = self.emb2out_model(enc) 


            logits = torch.sum((_topk_values.unsqueeze(-1) * logits).reshape(B, K, -1), dim=1)
            return logits

        def _predictor_path():
            # ensure first token isn't all-zeros
            topk[:, 0, :] = torch.where(
                topk[:, 0, :].sum(dim=-1, keepdim=True) == 0,
                self.random_token_emb.expand(B, E),
                topk[:, 0, :],
            )
            enc = self.predictor(topk, topk_times)
            logits = self.emb2out_model(enc) 
            return logits

        if self.training:
            if self.training and self.mode == "retriever":
                logits = _retriever_path()
            else:
                logits = _predictor_path()
        else:
            logits = _predictor_path()

        return logits



class REMedWithGenHPF(nn.Module):
    def __init__(
        self,
        genhpf_encoder: nn.Module,
        pred_dim: int = 512,
        num_classes: int = 1,
        pred_time: int = 48,
        max_retrieve_len: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = genhpf_encoder
        self.freeze_encoder = freeze_encoder
        self.num_classes = num_classes

        # encoder output dim -> pred_dim (fixes agg_embed_dim=128 vs pred_dim=512 mismatch)
        enc_out_dim = getattr(genhpf_encoder, "agg_embed_dim", None)
        if enc_out_dim is None:
            raise ValueError("genhpf_encoder must expose agg_embed_dim (encoder output dim).")

        self.enc_to_pred = nn.Identity() if enc_out_dim == pred_dim else nn.Linear(enc_out_dim, pred_dim)

        self.remed = REMed(
            pred_dim=pred_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_retrieve_len=max_retrieve_len,
            pred_time=pred_time,
            num_classes=num_classes,
        )

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()  # no dropout in encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        times: torch.Tensor,
        return_probs: bool = False,
    ):
        # keep encoder deterministic if frozen (Lightning may flip modules to train())
        if self.freeze_encoder:
            self.encoder.eval()

        reprs, pad_mask = self.encoder(input_ids, padding_mask=padding_mask)  # (B,S,enc_out_dim)
        reprs = self.enc_to_pred(reprs)  # (B,S,pred_dim)

        if pad_mask is not None and pad_mask.any():
            reprs = reprs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        logits = self.remed(reprs, times.float())  # (B,C)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # (B,)
            return (logits, torch.sigmoid(logits)) if return_probs else logits

        return (logits, torch.softmax(logits, dim=-1)) if return_probs else logits
    

class REMedLightningModule(lt.LightningModule):

    def __init__(
        self,
        model, 
        lr: float = 1e-5,
        wd: float = 1e-3,
        max_epochs: int = 100,
        pos_weight: float = 1.0,
        freeze_encoder: bool = True,
        use_warmup: bool = False,
        warmup_steps: int = 500,
        num_classes: int = 1,
        prediction_csv_path: str = '.'  
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.automatic_optimization = False

        self.num_classes = num_classes

        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
            )
            self.train_auroc = BinaryAUROC()
            self.train_auprc = BinaryAveragePrecision()
            self.val_auroc = BinaryAUROC()
            self.val_auprc = BinaryAveragePrecision()
            self.test_auroc = BinaryAUROC()
            self.test_auprc = BinaryAveragePrecision()
        else:
            self.criterion = nn.CrossEntropyLoss()
            # keep metrics out for multiclass unless you explicitly want them
        self.prediction_csv_path = prediction_csv_path
        self.train_step_preds = []
        self.train_step_labels = []
        self.val_step_preds = []
        self.val_step_labels = []
        self.test_step_preds = []
        self.test_step_logits = [] 
        self.test_step_labels = []
        self.test_step_subject_ids = []
        self.test_step_hadm_ids = []
        self.test_step_icustay_ids = []

        self._freeze_encoder = freeze_encoder

        # IMPORTANT: don't leave the model stuck in "predictor" mode from debug prints
        self.model.remed.set_mode("retriever")

    def on_train_batch_start(self, batch, batch_idx):
        if self._freeze_encoder and hasattr(self.model, "encoder"):
            self.model.encoder.eval()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            padding_mask=batch["padding_mask"],
            times=batch["times"],
        )

    def _step_once(self, batch, y, mode: str):
        opt = self.optimizers()
        # sch = self.lr_schedulers()

        self.model.remed.set_mode(mode)

        logits = self.forward(batch)

        if self.num_classes == 1:
            logits = logits.view(-1)
            loss = self.criterion(logits, y)
        else:
            
            logits = logits.view(y.size(0), -1)
            loss = self.criterion(logits, y.long())

        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()
        # if sch is not None:
        #     sch.step()

        return loss, logits

    def training_step(self, batch, batch_idx):
        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
        else:
            y = batch["label"].view(-1)

        loss1, _ = self._step_once(batch, y, mode="retriever")
        loss2, logits2 = self._step_once(batch, y, mode="predictor")

        loss = 0.5 * (loss1 + loss2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.num_classes == 1:
            probs = torch.sigmoid(logits2.detach().view(-1))
            self.train_step_labels.append(y.detach())
            self.train_step_preds.append(probs.detach())

        return loss

    def on_train_epoch_end(self):
        if self.num_classes != 1:
            return

        y = torch.cat(self.train_step_labels).long()
        p = torch.cat(self.train_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_p = self.all_gather(p).reshape(-1)

        self.log("train_auroc", self.train_auroc(gathered_p, gathered_y), prog_bar=True, sync_dist=True)
        self.log("train_auprc", self.train_auprc(gathered_p, gathered_y), prog_bar=True, sync_dist=True)

        self.train_step_labels.clear()
        self.train_step_preds.clear()

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        

    def validation_step(self, batch, batch_idx):
        self.model.remed.set_mode("predictor")

        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
            logits = self.forward(batch).view(-1)
            loss = self.criterion(logits, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            probs = torch.sigmoid(logits.detach())
            self.val_step_labels.append(y.detach())
            self.val_step_preds.append(probs.detach())
            return loss
        else:
            y = batch["label"].view(-1).long()
            logits = self.forward(batch).view(y.size(0), -1)
            loss = self.criterion(logits, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return loss

    def on_validation_epoch_end(self):
        if self.num_classes != 1:
            return

        y = torch.cat(self.val_step_labels).long()
        p = torch.cat(self.val_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_p = self.all_gather(p).reshape(-1)

        self.log("val_auroc", self.val_auroc(gathered_p, gathered_y), prog_bar=True, sync_dist=True)
        self.log("val_auprc", self.val_auprc(gathered_p, gathered_y), prog_bar=True, sync_dist=True)

        self.val_step_labels.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        self.model.remed.set_mode("predictor")

        if self.num_classes == 1:
            y = batch["label"].float().view(-1)
            logits = self.forward(batch).view(-1)
            loss = self.criterion(logits, y)
            self.log("test_loss", loss, prog_bar=True, on_epoch=True)
            probs = torch.sigmoid(logits.detach())

            self.test_step_logits.append(logits.detach())
            self.test_step_preds.append(probs.detach())
            self.test_step_labels.append(y.detach())

            self.test_step_subject_ids.append(batch["subject_id"])
            self.test_step_hadm_ids.append(batch["hadm_id"])
            self.test_step_icustay_ids.append(batch["icustay_id"])
            return loss
        else:
            y = batch["label"].view(-1).long()
            logits = self.forward(batch).view(y.size(0), -1)
            loss = self.criterion(logits, y)
            self.log("test_loss", loss, prog_bar=True, on_epoch=True)
            return loss

    def on_test_epoch_end(self):
        if self.num_classes != 1:
            return
        y = torch.cat(self.test_step_labels)
        pos_score = torch.cat(self.test_step_preds)
        logits = torch.cat(self.test_step_logits)
        subject_ids = torch.cat(self.test_step_subject_ids)
        hadm_ids = torch.cat(self.test_step_hadm_ids)
        icustay_ids = torch.cat(self.test_step_icustay_ids)

        gathered_y = self.all_gather(y)
        gathered_pos_score = self.all_gather(pos_score)
        gathered_logits = self.all_gather(logits)

        gathered_subject_ids = self.all_gather(subject_ids)
        gathered_hadm_ids = self.all_gather(hadm_ids)
        gathered_icustay_ids = self.all_gather(icustay_ids)

        gathered_y = gathered_y.reshape(-1)
        gathered_pos_score = gathered_pos_score.reshape(-1)
        gathered_logits = gathered_logits.reshape(-1)

        gathered_subject_ids = gathered_subject_ids.reshape(-1)
        gathered_hadm_ids = gathered_hadm_ids.reshape(-1)
        gathered_icustay_ids = gathered_icustay_ids.reshape(-1)

        auroc = self.test_auroc(gathered_pos_score,gathered_y.long())
        auprc = self.test_auprc(gathered_pos_score,gathered_y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True,)
        self.log("test_auprc", auprc, on_epoch=True,logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=gathered_y,
            y_score=gathered_pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3)

        if self.global_rank == 0:

            results = pd.DataFrame(
                {
                    "subject_id": gathered_subject_ids.cpu().numpy(),
                    "hadm_id": gathered_hadm_ids.cpu().numpy(),
                    "icustay_id": gathered_icustay_ids.cpu().numpy(),
                    "label": gathered_y.cpu().numpy(),
                    "prediction": gathered_pos_score.cpu().numpy(),
                    "logit": gathered_logits.cpu().numpy(),
                }
            )

            results.to_csv(self.prediction_csv_path, index=False)

        self.test_step_labels.clear()
        self.test_step_preds.clear()
        self.test_step_logits.clear()
        self.test_step_subject_ids.clear()
        self.test_step_hadm_ids.clear()
        self.test_step_icustay_ids.clear()

    # def configure_optimizers(self):

    #     opt = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    #     sch = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         opt,
    #         T_max=self.hparams.max_epochs,  # 75 epochs
    #         eta_min=0.0
    #     )

    #     return {
    #         "optimizer": opt,
    #         "lr_scheduler": {
    #             "scheduler": sch,
    #             "interval": "epoch",   # <-- key
    #             "frequency": 1
    #         }
    #     }
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if self.hparams.use_warmup:
            sch = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1 / 100, end_factor=1.0, total_iters=self.hparams.warmup_steps
            )
        else:
            sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=1)
        return {"optimizer": opt, "lr_scheduler": sch}
        # opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # if self.hparams.use_warmup:
        #     sch = torch.optim.lr_scheduler.LinearLR(
        #         opt, start_factor=1 / 100, end_factor=1.0, total_iters=self.hparams.warmup_steps
        #     )
        # else:
        #     sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=opt,
        #     T_max=75,
        #     eta_min=0)
        # return {"optimizer": opt, "lr_scheduler": sch}
    


#########################################################
# EHRMamba model
#########################################################

class EHRMambaNTPPretraining(lt.LightningModule):
    def __init__(
        self,
        cfg,
        backbone, 
        lr: float = 1e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        use_type: bool = True,
        use_visit: bool = True,
        use_stage: bool = True,
        use_time: bool = False,
        use_numeric: bool = False,
    ):
        
        
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.top_1_train = Accuracy(
            task="multiclass",
            num_classes=cfg.vocab_size,
            top_k=1,
            ignore_index=-100,
        )
        self.top_1_val = Accuracy(
            task="multiclass",
            num_classes=cfg.vocab_size,
            top_k=1,
            ignore_index=-100,
        )

        self.backbone = backbone(cfg)

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=cfg.vocab_size,
            embedding_size=cfg.hidden_size,
            pad_token_id=cfg.pad_token_id,
            type_vocab_size=cfg.type_vocab_size,
            visit_vocab_size=cfg.visit_vocab_size,
            stage_vocab_size=cfg.stage_vocab_size,
            dropout=dropout,

            use_type=use_type,
            use_visit=use_visit,
            use_stage=use_stage,

            use_position_embeddings=False,
            max_position_embeddings=0,

            use_time=use_time,
            time_in_features=1,
            time_out_features=16,

            use_numeric=use_numeric,
        )


        self.backbone.get_input_embeddings().weight = self.ehr_embeddings.tok_emb.weight
        self.backbone.get_output_embeddings().weight = self.ehr_embeddings.tok_emb.weight

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

    def forward(self, batch):

        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=batch["input_ids"],
            type_ids=batch.get("type_ids"),
            visit_ids=batch.get("visit_ids"),
            stage_ids=batch.get("stage_ids"),
            time_feats=batch.get("time_diff"),
            numeric_values=batch.get("numeric_values"),
            numeric_mask=batch.get("numeric_mask"),
        )

        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        loss = out.loss
        logits = out.logits              
        labels = batch["labels"]         

        preds  = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        target = labels[:,  1: ].contiguous().view(-1)

        top1 = self.top_1_train(preds, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top1", top1,  prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)

        loss = out.loss
        logits = out.logits              
        labels = batch["labels"]         

        preds  = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        target = labels[:,  1: ].contiguous().view(-1)

        top1 = self.top_1_train(preds, target)
        valid = (target != -100).sum()
        
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top1", top1,  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0,
            T_max=self.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
#########################################################
# HiBEHRT model
#########################################################


class HiBEHRT(nn.Module):
    def __init__(self, 
                 config,
                 backbone):
        super().__init__()
        self.config = config
        enc_cfg = copy.deepcopy(config)
        agg_cfg = copy.deepcopy(config)
        
        self.encoder = backbone(enc_cfg, add_pooling_layer=False)
        self.aggregator = backbone(agg_cfg, add_pooling_layer=False)

    def forward(
        self,
        inputs_embeds,        
        attention_mask,        
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        B, n, L, d = inputs_embeds.shape
        flat_embeds = inputs_embeds.reshape(B * n, L, d)       
        flat_mask   = attention_mask.reshape(B * n, L)         

        enc_out = self.encoder(
            inputs_embeds=flat_embeds,
            attention_mask=flat_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        enc_last = enc_out.last_hidden_state                      

        chunk_repr = enc_last[:, 0, :].reshape(B, n, d)            

        chunk_mask = (attention_mask.sum(dim=-1) > 0).long()       

        agg_out = self.aggregator(
            inputs_embeds=chunk_repr,
            attention_mask=chunk_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if return_dict:
            return BaseModelOutput(
                last_hidden_state=agg_out.last_hidden_state,       
                hidden_states=agg_out.hidden_states if output_hidden_states else None,
                attentions=None,
            )
        return (agg_out.last_hidden_state,)
    

class HiBEHRTModule(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,
        ckpt_path: str = None,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: str = 'cls',
        use_type: bool = False,
        use_visit: bool = False,
        use_stage: bool = False,
        use_numeric: bool = False,
        use_time: bool = False,
        optimizer: str = 'sgd',
        prediction_csv_path: str = '.'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.pooling = pooling
        self.backbone = HiBEHRT(config=config,backbone=backbone)
        self.optimizer = optimizer
        
        self.prediction_csv_path = prediction_csv_path
        self.train_step_preds = []
        self.train_step_labels = []
        self.val_step_preds = []
        self.val_step_labels = []
        self.test_step_preds = []
        self.test_step_logits = [] 
        self.test_step_labels = []
        self.test_step_subject_ids = []
        self.test_step_hadm_ids = []
        self.test_step_icustay_ids = []

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_type=use_type,
            use_visit=use_visit,
            use_stage=use_stage,
            use_time=use_time,
            use_numeric=use_numeric,
            use_position_embeddings=True,
            max_position_embeddings=(getattr(config, "max_position_embeddings", 512))
        )

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        if ckpt_path:
            self.get_pretrained_weights(ckpt_path=ckpt_path)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        
        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()


    def forward(
        self,
        input_ids,
        attention_mask,
        type_ids=None,
        visit_ids=None,
        stage_ids=None,
        time_feats=None,
        numeric_values=None,
        numeric_mask=None,
        labels=None,
    ):
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=input_ids,
            type_ids=type_ids,
            visit_ids=visit_ids,
            stage_ids=stage_ids,
            time_feats=time_feats,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
        )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  

        if self.pooling == 'mean':
            chunk_mask = (attention_mask.sum(dim=-1) > 0).type_as(last_hidden)  
            summed = (last_hidden * chunk_mask.unsqueeze(-1)).sum(dim=1)        
            lengths = chunk_mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)        
            pooled = summed / lengths                              
        elif self.pooling == 'cls':
            pooled = last_hidden[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)              
        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"] if self.hparams.use_type else None,
            visit_ids=batch["visit_ids"] if self.hparams.use_visit else None,
            stage_ids=batch["stage_ids"] if self.hparams.use_stage else None,
            time_feats=batch["time_diff"] if self.hparams.use_time else None,
            numeric_values=batch["numeric_values"] if self.hparams.use_numeric else None,
            numeric_mask=batch["numeric_mask"] if self.hparams.use_numeric else None,
        )

        y = batch["label"].float().view(-1)    
        loss = self.criterion(logits, y)

        pos_score = torch.sigmoid(logits)       

        self.train_step_labels.append(y.detach())
        self.train_step_preds.append(pos_score.detach())

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        y = torch.cat(self.train_step_labels)
        pos_score = torch.cat(self.train_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.train_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.train_auprc(gathered_pos_score, gathered_y.long())

        self.log('train_auroc', auroc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log('train_auprc', auprc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

        self.train_step_labels.clear()
        self.train_step_preds.clear()

    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"] if self.hparams.use_type else None,
            visit_ids=batch["visit_ids"] if self.hparams.use_visit else None,
            stage_ids=batch["stage_ids"] if self.hparams.use_stage else None,
            time_feats=batch["time_diff"] if self.hparams.use_time else None,
            numeric_values=batch["numeric_values"] if self.hparams.use_numeric else None,
            numeric_mask=batch["numeric_mask"] if self.hparams.use_numeric else None,
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.val_step_labels.append(y.detach())
        self.val_step_preds.append(pos_score.detach())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self,*arg, **kwargs) -> None:
        y = torch.cat(self.val_step_labels)
        pos_score = torch.cat(self.val_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_pos_score = self.all_gather(pos_score).reshape(-1)

        auroc = self.val_auroc(gathered_pos_score, gathered_y.long())
        auprc = self.val_auprc(gathered_pos_score, gathered_y.long())

        self.log('val_auroc', auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_auprc', auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.val_step_labels.clear()
        self.val_step_preds.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"] if self.hparams.use_type else None,
            visit_ids=batch["visit_ids"] if self.hparams.use_visit else None,
            stage_ids=batch["stage_ids"] if self.hparams.use_stage else None,
            time_feats=batch["time_diff"] if self.hparams.use_time else None,
            numeric_values=batch["numeric_values"] if self.hparams.use_numeric else None,
            numeric_mask=batch["numeric_mask"] if self.hparams.use_numeric else None,
        )

        y = batch["label"].float().view(-1)
        loss = self.criterion(logits, y)
        pos_score = torch.sigmoid(logits)

        self.test_step_logits.append(logits.detach())
        self.test_step_preds.append(pos_score.detach())
        self.test_step_labels.append(y.detach())

        self.test_step_subject_ids.append(batch["subject_id"])
        self.test_step_hadm_ids.append(batch["hadm_id"])
        self.test_step_icustay_ids.append(batch["icustay_id"])

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss    

    def on_test_epoch_end(self,*arg, **kwargs) -> None:
        y = torch.cat(self.test_step_labels)
        pos_score = torch.cat(self.test_step_preds)
        logits = torch.cat(self.test_step_logits)
        subject_ids = torch.cat(self.test_step_subject_ids)
        hadm_ids = torch.cat(self.test_step_hadm_ids)
        icustay_ids = torch.cat(self.test_step_icustay_ids)

        gathered_y = self.all_gather(y)
        gathered_pos_score = self.all_gather(pos_score)
        gathered_logits = self.all_gather(logits)

        gathered_subject_ids = self.all_gather(subject_ids)
        gathered_hadm_ids = self.all_gather(hadm_ids)
        gathered_icustay_ids = self.all_gather(icustay_ids)

        gathered_y = gathered_y.reshape(-1)
        gathered_pos_score = gathered_pos_score.reshape(-1)
        gathered_logits = gathered_logits.reshape(-1)

        gathered_subject_ids = gathered_subject_ids.reshape(-1)
        gathered_hadm_ids = gathered_hadm_ids.reshape(-1)
        gathered_icustay_ids = gathered_icustay_ids.reshape(-1)

        auroc = self.test_auroc(gathered_pos_score,gathered_y.long())
        auprc = self.test_auprc(gathered_pos_score,gathered_y.long())

        self.log("test_auroc", auroc, on_epoch=True, logger=True,)
        self.log("test_auprc", auprc, on_epoch=True,logger=True)

        log_bootstrap_ci_text_percentile(
            module=self,
            y_true=gathered_y,
            y_score=gathered_pos_score,
            prefix="test",
            num_iter=1000,
            alpha=0.05,
            ndigits=3)

        if self.global_rank == 0:

            results = pd.DataFrame(
                {
                    "subject_id": gathered_subject_ids.cpu().numpy(),
                    "hadm_id": gathered_hadm_ids.cpu().numpy(),
                    "icustay_id": gathered_icustay_ids.cpu().numpy(),
                    "label": gathered_y.cpu().numpy(),
                    "prediction": gathered_pos_score.cpu().numpy(),
                    "logit": gathered_logits.cpu().numpy(),
                }
            )

            results.to_csv(self.prediction_csv_path, index=False)

        self.test_step_labels.clear()
        self.test_step_preds.clear()
        self.test_step_logits.clear()
        self.test_step_subject_ids.clear()
        self.test_step_hadm_ids.clear()
        self.test_step_icustay_ids.clear()

    def configure_optimizers(self):

        if self.optimizer == 'adamw':
            decay, no_decay = [], []
            for name, p in self.named_parameters():
                if "bias" in name or "LayerNorm" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)

            optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": self.wd},
                                        {"params": no_decay, "weight_decay": 0.0}],
                                        lr=self.lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9,
                                        nesterov=True)


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0)

        return {"optimizer": optimizer,"lr_scheduler": scheduler,}
    
    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.",
                         "backbone.lm_head.", "classifier.", "criterion."]
        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                remapped[k] = v
                continue

            if k.startswith("backbone.bert."):
                rest = k[len("backbone.bert."):]
                remapped["backbone.encoder." + rest] = v
                remapped["backbone.aggregator." + rest] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)


#########################################################
# LLM baselines
#########################################################


def load_hf_model(model_name: str, hf_token: str = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()
   

    yes_token_id = None
    no_token_id = None

    for candidate in ["Yes", " Yes", "yes", " yes"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            yes_token_id = ids[0]
            break

    for candidate in ["No", " No", "no", " no"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            no_token_id = ids[0]
            break

    if yes_token_id is None:
        raise ValueError("Could not find a single-token encoding for 'Yes'")
    if no_token_id is None:
        raise ValueError("Could not find a single-token encoding for 'No'")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
    }