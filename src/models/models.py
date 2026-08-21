
import os
import torch

import pandas as pd
import lightning as lt
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Mapping
from .utils import Time2Vec, log_bootstrap_ci_text_percentile
from torchmetrics.classification import Accuracy, BinaryAUROC, BinaryAveragePrecision





class EHREmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        type_vocab_size: int = 28,
        visit_vocab_size: int = 102,
        stage_vocab_size: int = 5,
        dropout: float = 0.1,
        use_position_embeddings: bool = False,
        max_position_embeddings: int = 0,
        use_time: bool = False,
        time_in_features: int = 1,
        time_out_features: int = 16,
        use_numeric: bool = False,
        numeric_hidden_size: int = 16,
        use_type: bool = True,
        use_visit: bool = True,
        use_stage: bool = True,
    ):
        super().__init__()
        
        self.use_type = use_type
        self.use_visit = use_visit
        self.use_stage = use_stage

        self.tok_emb   = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        
        if use_type:
            self.type_emb  = nn.Embedding(type_vocab_size,  embedding_size, padding_idx=pad_token_id)
        else:
            self.type_emb = None
        if use_visit:    
            self.visit_emb = nn.Embedding(visit_vocab_size, embedding_size, padding_idx=pad_token_id)
        else:
            self.visit_emb = None
        if use_stage:
            self.stage_emb = nn.Embedding(stage_vocab_size, embedding_size, padding_idx=pad_token_id)
        else:
            self.stage_emb = None

        self.use_position_embeddings = use_position_embeddings
        
        if use_position_embeddings:
            if max_position_embeddings <= 0:
                raise ValueError("max_position_embeddings must be > 0 when use_position_embeddings=True")
            self.pos_emb = nn.Embedding(max_position_embeddings, embedding_size)
        else:
            self.pos_emb = None

        
        self.use_time = use_time
        
        if use_time:
            self.time2vec = Time2Vec(
                in_features=time_in_features,
                out_features=time_out_features,
                periodic_activation=torch.sin,
            )
            self.time_proj = nn.Linear(time_out_features, embedding_size)
        else:
            self.time2vec = None
            self.time_proj = None

        
        self.use_numeric = use_numeric
        
        if use_numeric:
            self.numeric_hidden_size = numeric_hidden_size
            
            self.num_proj1 = nn.Linear(1, numeric_hidden_size)
            self.num_proj2 = nn.Linear(numeric_hidden_size, embedding_size)
            self.num_act = nn.GELU()

            
            self.null_numeric = nn.Parameter(torch.zeros(embedding_size))
            nn.init.normal_(self.null_numeric, mean=0.0, std=0.02)

            nn.init.xavier_uniform_(self.num_proj1.weight)
            nn.init.zeros_(self.num_proj1.bias)
            nn.init.xavier_uniform_(self.num_proj2.weight)
            nn.init.zeros_(self.num_proj2.bias)
        else:
            self.num_proj1 = None
            self.num_proj2 = None
            self.num_act = None
            self.null_numeric = None

        self.norm = nn.LayerNorm(embedding_size)
        self.drop = nn.Dropout(dropout)

    def encode(
        self,
        input_ids,
        type_ids=None,
        visit_ids=None,
        stage_ids=None,
        time_feats=None,          
        numeric_values=None,     
        numeric_mask=None,        
    ):
        
        x = self.tok_emb(input_ids.long())
        if self.use_type:
            if type_ids is None:
                raise ValueError("type_ids must be provided when use_type=True")
            x = x + self.type_emb(type_ids.long())
            
        if self.use_visit:
            if visit_ids is None:
                raise ValueError("visit_ids must be provided when use_visit=True")
            x = x + self.visit_emb(visit_ids.long())
            
        if self.use_stage:
            if stage_ids is None:
                raise ValueError("stage_ids must be provided when use_stage=True")
            x = x + self.stage_emb(stage_ids.long())

        
        if self.pos_emb is not None:
            shape = input_ids.size()
            seqlen = shape[-1]

            position_ids = torch.arange(seqlen, device=input_ids.device)

            if input_ids.dim() == 2:        
                bsz = shape[0]
                position_ids = position_ids.unsqueeze(0).expand(bsz, seqlen)          
            elif input_ids.dim() == 3:        
                bsz, n = shape[0], shape[1]
                position_ids = position_ids.view(1, 1, seqlen).expand(bsz, n, seqlen) 
            else:
                raise ValueError(f"Unsupported input_ids.dim()={input_ids.dim()}")
            x = x + self.pos_emb(position_ids)

        
        if self.use_time:
            if time_feats is None:
                raise ValueError("time_feats must be provided when use_time=True")
            if time_feats.dim() == 2:
                time_feats = time_feats.unsqueeze(-1)
            elif time_feats.dim() != 3:
                raise ValueError(f"Unexpected time_feats.dim()={time_feats.dim()}, expected 2 or 3")
            t = self.time2vec(time_feats.float())   
            t = self.time_proj(t)                   
            x = x + t

        
        if self.use_numeric:
            if numeric_values is None or numeric_mask is None:
                raise ValueError("numeric_values and numeric_mask must be provided when use_numeric=True")

            
            v = numeric_values.float().unsqueeze(-1)       
            
            h = self.num_act(self.num_proj1(v))             
            num_emb = self.num_proj2(h)                     

            mask = numeric_mask.bool().unsqueeze(-1)        
            num_emb = torch.where(mask, num_emb, self.null_numeric.view(1, 1, -1))
            x = x + num_emb

        return self.drop(self.norm(x))

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        return self.encode(
            input_ids=input_ids,
            type_ids=kwargs.get("type_ids"),
            visit_ids=kwargs.get("visit_ids"),
            stage_ids=kwargs.get("stage_ids"),
            time_feats=kwargs.get("time_feats"),
            numeric_values=kwargs.get("numeric_values"),
            numeric_mask=kwargs.get("numeric_mask"),
        )
    

class MLMPretraining(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,                      # e.g. ModernBertForMaskedLM, BertForMaskedLM, RoFormerForMaskedLM
        lr: float = 1e-6,
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
            num_classes=config.vocab_size,
            top_k=1,
            ignore_index=-100,
        )
        self.top_1_val = Accuracy(
            task="multiclass",
            num_classes=config.vocab_size,
            top_k=1,
            ignore_index=-100,
        )

        # instantiate backbone
        self.backbone = backbone(config)

        # ---- RoPE vs non-RoPE auto-detection ----
        rope_model_types = {"modernbert", "roformer"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        # EHREmbeddings: no pos-emb for RoPE, local pos-emb for others
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
            use_position_embeddings=not is_rope,
            max_position_embeddings=(
                getattr(config, "max_position_embeddings", 0)
                if not is_rope
                else 0
            ),
            
        )

        # tie LM head to token embeddings
        lm_head = self.backbone.get_output_embeddings()
        lm_head.weight = self.ehr_embeddings.tok_emb.weight

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

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

        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            time_feats=batch.get("time_diff", None),
            numeric_values=batch.get("numeric_values", None),
            numeric_mask=batch.get("numeric_mask", None),
            labels=batch["labels"],
        )

        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)

        top1 = self.top_1_train(preds, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_top1", top1,  prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            visit_ids=batch["visit_ids"],
            stage_ids=batch["stage_ids"],
            time_feats=batch.get("time_diff", None),
            numeric_values=batch.get("numeric_values", None),
            numeric_mask=batch.get("numeric_mask", None),
            labels=batch["labels"],
        )

        loss = out.loss
        preds  = out.logits.view(-1, out.logits.size(-1))
        target = batch["labels"].view(-1)

        top1 = self.top_1_val(preds, target)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_top1", top1,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0,
            T_max=self.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    

class EvalModel(lt.LightningModule):
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
        self.save_hyperparameters(ignore=["backbone"])
        self.pooling = pooling
        self.backbone = backbone(config)
        self.optimizer = optimizer
        rope_model_types = {"modernbert", "roformer", "mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

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
            use_position_embeddings=not is_rope,
            max_position_embeddings=(
                getattr(config, "max_position_embeddings", 0)
                if not is_rope
                else 0
            ),
            use_time=use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=use_numeric,  
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

        # metrics (unchanged)
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

        # pooling
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  
            summed = (last_hidden * mask).sum(dim=1)                 
            lengths = mask.sum(dim=1).clamp(min=1.0)                  
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
            labels=None,
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
            labels=None,
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
            labels=None,
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

    def on_test_epoch_end(self, *arg, **kwargs) -> None:

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
                # biases + LayerNorm weights should NOT get weight decay
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
            optimizer = torch.optim.Adam(self.parameters(),
                                        lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=self.wd,)


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0)

        return {"optimizer": optimizer,"lr_scheduler": scheduler,}
    
    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        mt = getattr(self.backbone.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone.",
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            else:
                new_k = k

            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)




###############################
# Ours
###############################

class CLMBREncoders(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        batch_processor=None,
        normalize: bool = False,
        freeze: bool = False,
        device: str = None,
    ):
        super().__init__()

        self.model = model
        self.batch_processor = batch_processor
        self.normalize = normalize
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device_)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        batch: Dict[str, Any],
        query_key: str = "query",
        history_key: str = "history",
    ) -> Dict[str, torch.Tensor]:

        q_batch = batch[query_key]
        h_batch = batch[history_key]

        q_vec = self._encode_packed_batch(q_batch)  # [B, H]

        h_vec_flat = self._encode_packed_batch(h_batch)  # [B*K, H]

        B, K = batch["history_valid_mask"].shape
        H = h_vec_flat.shape[-1]

        h_vec = h_vec_flat.reshape(B, K, H)

        return {
            "query_vec": q_vec,
            "hist_vec": h_vec,
        }

    def _encode_packed_batch(
        self,
        raw_batch: Mapping[str, Any],
    ) -> torch.Tensor:
#         if self.batch_processor is not None:
#             batch = self.batch_processor.collate([raw_batch])["batch"]
#             batch = femr.models.transformer.remove_first_dimension(batch)
#         else:
        batch = raw_batch

        transformer_batch = self._move_transformer_to_device(batch["transformer"])

        reprs = self.model.transformer(transformer_batch)

        patient_lengths = torch.as_tensor(
            transformer_batch["patient_lengths"],
            dtype=torch.long,
            device=reprs.device,
        )

        end_positions = torch.cumsum(patient_lengths, dim=0) - 1

        vec = reprs[end_positions]

        if self.normalize:
            vec = nn.functional.normalize(vec, p=2, dim=-1)

        return vec

    def _move_transformer_to_device(
        self,
        transformer_batch: Mapping[str, Any],
    ) -> Dict[str, Any]:

        out = {}

        for k, v in transformer_batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device_)
            else:
                out[k] = torch.as_tensor(v).to(self.device_)

        return out


class EHRRAPEncoders(nn.Module):
    def __init__(
        self,
        config,
        backbone,                 
        dropout: float = 0.1,
        pooling: str = "cls",       
        use_time: bool = False,
        use_numeric: bool = False,
        ckpt_path = None,
        return_token_level: bool = False,
    ):
        super().__init__()
        self.config = config
        self.pooling = pooling
        self.use_time = use_time
        self.use_numeric = use_numeric
        self.return_token_level = return_token_level
        self.backbone = backbone(config)

        rope_model_types = {"modernbert", "roformer","mamba"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types

        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(getattr(config, "max_position_embeddings", 0) if not is_rope else 0),
            use_time=use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=use_numeric,
        )

        
        if ckpt_path:
            self.get_pretrained_weights(ckpt_path=ckpt_path)


    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,L,1]
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return summed/lengths

    def _encode(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=x["input_ids"],
            type_ids=x["type_ids"],
            visit_ids=x["visit_ids"],
            stage_ids=x["stage_ids"],
            time_feats=x.get("time_diff", None) if self.use_time else None,
            numeric_values=x.get("numeric_values", None) if self.use_numeric else None,
            numeric_mask=x.get("numeric_mask", None) if self.use_numeric else None)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=x["attention_mask"],
            output_hidden_states=False,
            return_dict=True)
        
        seq = out.last_hidden_state
        vec = self._pool(seq, x["attention_mask"])
        return {"seq": seq, "vec": vec}

    def _flatten_history(self, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, K, L = h["input_ids"].shape
        flat = {}
        for k, v in h.items():
            if v is None:
                continue
            if v.dim() == 3:
                flat[k] = v.reshape(B * K, L)
        return flat

    def _unflatten_history(
        self,
        seq_flat: torch.Tensor,
        vec_flat: torch.Tensor, 
        B: int,
        K: int,
    ) -> Dict[str, torch.Tensor]:
        
        L = seq_flat.shape[1]
        H = seq_flat.shape[2]
        seq = seq_flat.reshape(B, K, L, H)
        vec = vec_flat.reshape(B, K, H)
        return {"seq": seq, "vec": vec}



    def forward(
        self,
        batch: Dict[str, Any],
        query_key: str = "query",
        history_key: str = "history", 
    ) -> Dict[str, torch.Tensor]:
        q = batch[query_key]
        h = batch[history_key]

        q_out = self._encode(q)

        B, K, L = h["input_ids"].shape
        h_flat = self._flatten_history(h)
        h_out_flat = self._encode(h_flat)
        h_out = self._unflatten_history(h_out_flat["seq"], h_out_flat["vec"], B=B, K=K)

        out = {"query_vec": q_out["vec"], "hist_vec": h_out["vec"]}

        if self.return_token_level:
            out["query_seq"]  = q_out["seq"]
            out["hist_seq"]   = h_out["seq"]
            out["query_mask"] = q["attention_mask"]
            out["hist_mask"]  = h["attention_mask"]

        return out
    
    def get_pretrained_weights(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        mt = getattr(self.backbone.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone.",
        }
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head."]

        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            else:
                new_k = k

            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("weights loaded successfully!")
        print("missing keys:", missing)
        print("+" * 50)
        print("unexpected keys:", unexpected)





class PrototypeRetrievalModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_prototypes: int = 256,
        query_temperature: float = 0.1,
        history_temperature: float = 0.025,
        normalize_prototypes: bool = True,
        softmax_threshold: float = 0.8,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.query_temperature = float(query_temperature)
        self.history_temperature = float(history_temperature)
        self.normalize_prototypes = normalize_prototypes
    
        self.softmax_threshold = float(softmax_threshold)
        self.softmax_temperature = float(softmax_temperature)

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)


    def _proto_probs(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        P = self.prototypes
        x = F.normalize(x, p=2, dim=-1)
        P = F.normalize(P, p=2, dim=-1)
        logits = x @ P.t()
        return F.softmax(logits / temperature, dim=-1)

    def _compute_neg_ce(self, q_probs: torch.Tensor, h_probs: torch.Tensor) -> torch.Tensor:
        return (q_probs.unsqueeze(1) * h_probs.clamp_min(1e-8).log()).sum(dim=-1)



    def _weights_and_mask_softmax(
        self,
        scores: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if valid_mask is not None:
            valid_bool = valid_mask.bool()
            scores = scores.masked_fill(~valid_bool, -1e9)
        else:
            valid_bool = torch.ones_like(scores, dtype=torch.bool)

        weights = F.softmax(scores / self.softmax_temperature, dim=-1)

        hard = (weights >= self.softmax_threshold).float()
        keep = hard + weights - weights.detach()

        keep = keep * valid_bool.float()

        return {
            "attn_weights": weights,                  
            "attn_mask": keep,    
        }

    def _entropy(self, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def forward(
        self,
        query_vec: torch.Tensor,
        hist_vec: torch.Tensor,
        hist_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, K, _ = hist_vec.shape

        if hist_valid_mask is not None:
            hist_valid_mask = hist_valid_mask.to(device=hist_vec.device, dtype=torch.long)
            valid_f = hist_valid_mask.to(dtype=hist_vec.dtype)
            n_valid = valid_f.sum().clamp_min(1.0)
        else:
            valid_f = None
            n_valid = None

        q_probs = self._proto_probs(query_vec,temperature= self.query_temperature )   # [B, P]
        h_probs = self._proto_probs(hist_vec, temperature= self.history_temperature)    # [B, K, P]

        q_ent_all = self._entropy(q_probs)
        h_ent_all = self._entropy(h_probs)

        q_max = q_probs.max(dim=-1).values.mean()
        h_max_all = h_probs.max(dim=-1).values

        if valid_f is None:
            q_ent = q_ent_all.mean()
            h_ent = h_ent_all.mean()
            h_max = h_max_all.mean()
        else:
            q_ent = q_ent_all.mean()
            h_ent = (h_ent_all * valid_f).sum() / n_valid
            h_max = (h_max_all * valid_f).sum() / n_valid

        mean_q_probs = q_probs.mean(dim=0)
        if valid_f is None:
            mean_h_probs = h_probs.mean(dim=(0, 1))
        else:
            valid = valid_f.unsqueeze(-1)
            mean_h_probs = (h_probs * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)

        mean_q_ent = self._entropy(mean_q_probs)
        mean_h_ent = self._entropy(mean_h_probs)

        neg_ce = self._compute_neg_ce(q_probs, h_probs)   # [B, K]

        wm = self._weights_and_mask_softmax(neg_ce, valid_mask=hist_valid_mask)

        out = {
            "neg_ce": neg_ce,
            "attn_mask": wm["attn_mask"],
            "attn_weights": wm["attn_weights"],
            "query_probs": q_probs,
            "hist_probs": h_probs,
            "diag_q_ent": q_ent,
            "diag_q_max": q_max,
            "diag_h_ent": h_ent,
            "diag_h_max": h_max,
            "diag_mean_q_ent": mean_q_ent,
            "diag_mean_h_ent": mean_h_ent,
        }
        return out
    



class FusionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_weights_as_gating: bool = False,
        output_mode: str = "query",  
        return_seq: bool = False,
    ):
        super().__init__()
        self.use_weights_as_gating = use_weights_as_gating
        self.output_mode = output_mode
        self.return_seq = return_seq

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_mult * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True, 
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,)

        self.out_norm = nn.LayerNorm(hidden_size)
        # self.hist_score_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_vec: torch.Tensor,
        hist_vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
        hist_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B, K, H = hist_vec.shape
        assert query_vec.shape == (B, H)

        if self.use_weights_as_gating and attn_weights is not None:
            hist_vec = hist_vec * attn_weights.unsqueeze(-1).to(hist_vec.dtype)
            # hist_vec = self.hist_score_proj(hist_vec)

        # if self.use_weights_as_gating and attn_weights is not None:
        #     alpha = attn_weights.unsqueeze(-1).to(hist_vec.dtype)
        #     hist_vec = hist_vec * (1.0 + alpha)


        # if self.use_weights_as_gating and attn_weights is not None:
        #     mask = hist_valid_mask.float()                         # [B, K]
        #     num_valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]

        #     scaled_weights = attn_weights * num_valid              # rescale
        #     scaled_weights = scaled_weights * mask                 # zero out padding

        #     hist_vec = hist_vec * scaled_weights.unsqueeze(-1).to(hist_vec.dtype)

        elif attn_mask is not None:
            hist_vec = hist_vec * attn_mask.unsqueeze(-1).to(hist_vec.dtype)

        x = torch.cat([query_vec.unsqueeze(1), hist_vec], dim=1)  # [B, 1+K, H]

        if hist_valid_mask is None:
            src_key_padding_mask = None
            keep = torch.ones(B, 1 + K, device=x.device, dtype=x.dtype)
        else:
            query_valid = torch.ones(B, 1, device=x.device, dtype=torch.bool)
            keep_bool = torch.cat([query_valid, hist_valid_mask.bool()], dim=1)   # [B, 1+K]
            src_key_padding_mask = ~keep_bool
            keep = keep_bool.to(x.dtype)

        x_fused = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x_fused = self.out_norm(x_fused)

        if self.output_mode == "query":
            fused_vec = x_fused[:, 0, :]
        elif self.output_mode == "mean":
            w = keep.unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            fused_vec = (x_fused * w).sum(dim=1) / denom

        out = {"fused_vec": fused_vec}
        if self.return_seq:
            out["fused_seq"] = x_fused
            out["fused_keep_mask"] = keep
        return out
    


class EHRRAPEvalModel(lt.LightningModule):
    def __init__(
        self,
        config,
        backbone,
        ckpt_path: Optional[str] = None,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: str = "cls",
        use_numeric: bool = False,
        use_time: bool = False,
        # --- prototype module ---
        num_prototypes: int = 256,
        query_temperature: float = 0.1,
        history_temperature: float = 0.025,
        normalize_prototypes: bool = True,
        softmax_threshold: float = 0.8,
        softmax_temperature: float = 1.0,
        sample_ent_lambda: float = 0.001,
        usage_ent_lambda: float = 0.001,
        use_prototypes: bool = False,
        # --- fusion module ---
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        fusion_ff_mult: int = 4,
        fusion_output_mode: str = "query",
        use_weights_as_gating: bool = False,
        prediction_csv_path: str = '.'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.config = config
        self.sample_ent_lambda = float(sample_ent_lambda)
        self.usage_ent_lambda = float(usage_ent_lambda)
        self.use_prototypes = use_prototypes
        self.encoders = EHRRAPEncoders(
            config=config,
            backbone=backbone,
            dropout=dropout,
            pooling=pooling,
            use_time=use_time,
            use_numeric=use_numeric,
            ckpt_path=ckpt_path,
            return_token_level=False,
        )

        self.prototypes = PrototypeRetrievalModule(
            hidden_size=config.hidden_size,
            num_prototypes=num_prototypes,
            query_temperature=query_temperature,
            history_temperature=history_temperature,
            normalize_prototypes=normalize_prototypes,
            softmax_threshold=softmax_threshold,
            softmax_temperature=softmax_temperature,
        )

        self.fusion = FusionModule(
            hidden_size=config.hidden_size,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            ff_mult=fusion_ff_mult,
            dropout=dropout,
            use_weights_as_gating=use_weights_as_gating,
            output_mode=fusion_output_mode,
            return_seq=False,
        )

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

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
        
#         if freeze:
#             for p in self.encoders.parameters():
#                 p.requires_grad = False
#             for p in self.prototypes.parameters():
#                 p.requires_grad = True
#             for p in self.fusion.parameters():
#                 p.requires_grad = True
#             for p in self.classifier.parameters():
#                 p.requires_grad = True

# with ordering
    # def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    #     enc_out = self.encoders(batch, query_key="query", history_key="history")

    #     proto_out = None
    #     attn_mask = None
    #     attn_weights = None
    #     hist_valid_mask = batch["history_valid_mask"]
    #     hist_vec = enc_out["hist_vec"]

    #     if self.use_prototypes:
    #         proto_out = self.prototypes(
    #             query_vec=enc_out["query_vec"],
    #             hist_vec=hist_vec,
    #             hist_valid_mask=hist_valid_mask,
    #         )

    #         attn_mask = proto_out["attn_mask"]
    #         attn_weights = proto_out["attn_weights"]

    #         # sort by prototype score, highest first
    #         sort_scores = proto_out["attn_weights"]  # or proto_out["neg_ce"]
    #         sort_idx = torch.argsort(sort_scores, dim=1, descending=True)

    #         hist_vec = torch.gather(
    #             hist_vec,
    #             dim=1,
    #             index=sort_idx.unsqueeze(-1).expand(-1, -1, hist_vec.size(-1)),
    #         )

    #         hist_valid_mask = torch.gather(hist_valid_mask, dim=1, index=sort_idx)

    #         if attn_mask is not None:
    #             attn_mask = torch.gather(attn_mask, dim=1, index=sort_idx)

    #         if attn_weights is not None:
    #             attn_weights = torch.gather(attn_weights, dim=1, index=sort_idx)

    #     fuse_out = self.fusion(
    #         query_vec=enc_out["query_vec"],
    #         hist_vec=hist_vec,
    #         attn_mask=attn_mask,
    #         attn_weights=attn_weights,
    #         hist_valid_mask=hist_valid_mask,
    #     )

    #     fused_vec = fuse_out["fused_vec"]
    #     logits = self.classifier(fused_vec).squeeze(-1)

    #     out = {
    #         "logits": logits,
    #         "fused_vec": fused_vec,
    #     }
    #     if proto_out is not None:
    #         out["proto"] = proto_out

    #     return out

# without ordering
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        enc_out = self.encoders(batch, query_key="query", history_key="history")

        proto_out = None
        attn_mask = None
        attn_weights = None
        hist_valid_mask = batch["history_valid_mask"]

        if self.use_prototypes:
            proto_out = self.prototypes(
                query_vec=enc_out["query_vec"],
                hist_vec=enc_out["hist_vec"],
                hist_valid_mask=hist_valid_mask,
            )
            attn_mask = proto_out["attn_mask"]          # soft/ST mask
            attn_weights = proto_out["attn_weights"]    # soft weights

        fuse_out = self.fusion(
            query_vec=enc_out["query_vec"],
            hist_vec=enc_out["hist_vec"],
            attn_mask=attn_mask,
            attn_weights=attn_weights,
            hist_valid_mask=hist_valid_mask,            # real padding mask only
        )

        fused_vec = fuse_out["fused_vec"]
        logits = self.classifier(fused_vec).squeeze(-1)

        out = {
            "logits": logits,
            "fused_vec": fused_vec,
        }
        if proto_out is not None:
            out["proto"] = proto_out

        return out

    def shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        out = self.forward(batch)

        proto = out.get("proto", None)

        # --- log proto diagnostics if present ---
        if proto is not None and "diag_q_ent" in proto:
            self.log(f"{stage}_sample_q_ent", proto["diag_q_ent"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_q_max", proto["diag_q_max"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_sample_h_ent", proto["diag_h_ent"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_h_max", proto["diag_h_max"], prog_bar=True, on_step=True if stage == 'train' else False, on_epoch=True, logger=True, sync_dist=True)

        if proto is not None and "diag_mean_q_ent" in proto:
            self.log(f"{stage}_batch_q_ent", proto["diag_mean_q_ent"], prog_bar=False, on_step=True if stage == "train" else False, on_epoch=True, logger=True, sync_dist=True)
            self.log(f"{stage}_batch_h_ent", proto["diag_mean_h_ent"], prog_bar=False, on_step=True if stage == "train" else False, on_epoch=True, logger=True, sync_dist=True)

        logits = out["logits"]
        y = batch["label"].float().view(-1)

        task_loss = self.criterion(logits, y)


        sample_ent_reg = task_loss.new_zeros(())
        usage_ent_reg = task_loss.new_zeros(())

        if proto is not None and "diag_q_ent" in proto and self.sample_ent_lambda > 0.0:
            sample_ent_reg = proto["diag_q_ent"] + proto["diag_h_ent"]
            # sample_ent_reg = proto["diag_h_ent"]
        if proto is not None and "diag_mean_q_ent" in proto and self.usage_ent_lambda > 0.0:
            usage_ent_reg = -(proto["diag_mean_q_ent"] + proto["diag_mean_h_ent"])
            # usage_ent_reg = -(proto["diag_mean_h_ent"])


        # total_loss = task_loss + self.ent_lambda * ent_reg
        total_loss = (
            task_loss
            + self.sample_ent_lambda * sample_ent_reg
            + self.usage_ent_lambda * usage_ent_reg
        )

        # --- logging losses ---
        self.log(f"{stage}_loss", task_loss, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{stage}_total_loss", total_loss, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)


        pos_score = torch.sigmoid(logits)

        if stage == "train":
            self.train_step_labels.append(y.detach())
            self.train_step_preds.append(pos_score.detach())
        elif stage == "val":
            self.val_step_labels.append(y.detach())
            self.val_step_preds.append(pos_score.detach())
        elif stage == "test":
            self.test_step_logits.append(logits.detach())
            self.test_step_preds.append(pos_score.detach())
            self.test_step_labels.append(y.detach())

            self.test_step_subject_ids.append(batch["subject_id"])
            self.test_step_hadm_ids.append(batch["hadm_id"])
            self.test_step_icustay_ids.append(batch["icustay_id"])

        if proto is not None and "attn_weights" in proto and proto["attn_weights"] is not None and not self.fusion.use_weights_as_gating:
            with torch.no_grad():
                weights = proto["attn_weights"].float()
                mask = batch["history_valid_mask"].float()

                hard_keep = (weights >= self.prototypes.softmax_threshold).float()
                keep_rate = (hard_keep * mask).sum() / mask.sum().clamp_min(1.0)

            self.log(f"{stage}_keep_rate",keep_rate,prog_bar=False, on_step=True, on_epoch=True,logger=True,sync_dist=True)

        if proto is not None and "attn_weights" in proto and proto["attn_weights"] is not None and self.fusion.use_weights_as_gating:
            with torch.no_grad():
                weights = proto["attn_weights"].float()
                mask = batch["history_valid_mask"].float()

                weights = weights * mask
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                weights_norm = weights / denom

                weight_entropy = -(weights_norm * (weights_norm.clamp_min(1e-8).log())).sum(dim=1)

                valid_counts = mask.sum(dim=1).clamp_min(1.0)
                max_entropy = valid_counts.log().clamp_min(1e-8)
                weight_entropy_norm = (weight_entropy / max_entropy).mean()

            self.log(f"{stage}_weight_entropy",weight_entropy.mean(), prog_bar=False, on_step=True, on_epoch=True,logger=True,sync_dist=True)
            self.log(f"{stage}_weight_entropy_norm", weight_entropy_norm, prog_bar=False, on_step=True, on_epoch=True, logger=True,sync_dist=True)


        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="test")


    # def on_train_start(self) -> None:
    #     if not self.use_prototypes:
    #         return
    #     self._proto_init = self.prototypes.prototypes.detach().clone()

    # def on_after_backward(self) -> None:
    #     if not self.use_prototypes or not hasattr(self, "_proto_init"):
    #         return

    #     current = self.prototypes.prototypes.detach()
    #     delta_norm = (current - self._proto_init).norm()
    #     current_norm = current.norm().clamp_min(1e-12)
    #     rel_delta = delta_norm / current_norm

        
    #     proto_param = self.prototypes.prototypes

    #     if proto_param.grad is None:
    #         grad_norm = 0.0
    #     else:
    #         grad_norm = proto_param.grad.norm()
            
    #     self.log(
    #         "train_proto_delta_norm",
    #         delta_norm,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )
    #     self.log(
    #         "train_proto_rel_delta",
    #         rel_delta,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )    

    #     self.log(
    #         "train_proto_grad_norm",
    #         grad_norm,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True)


    def on_train_epoch_end(self) -> None:
        if not self.train_step_preds:
            return

        y = torch.cat(self.train_step_labels)
        p = torch.cat(self.train_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_p = self.all_gather(p).reshape(-1)

        train_auroc = self.train_auroc(gathered_p, gathered_y.long())
        train_auprc = self.train_auprc(gathered_p, gathered_y.long())

        self.log("train_auroc", train_auroc, on_epoch=True, logger=True,prog_bar=True, sync_dist=True,)
        self.log("train_auprc", train_auprc, on_epoch=True, logger=True,prog_bar=True,sync_dist=True,)

        self.train_step_labels.clear()
        self.train_step_preds.clear()

    def on_validation_epoch_end(self) -> None:
        if not self.val_step_preds:
            return

        y = torch.cat(self.val_step_labels)
        p = torch.cat(self.val_step_preds)

        gathered_y = self.all_gather(y).reshape(-1)
        gathered_p = self.all_gather(p).reshape(-1)

        val_auroc = self.val_auroc(gathered_p, gathered_y.long())
        val_auprc = self.val_auprc(gathered_p, gathered_y.long())

        self.log("val_auroc", val_auroc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auprc", val_auprc, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.val_step_labels.clear()
        self.val_step_preds.clear()

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
        param_groups = [
            {"params": self.encoders.parameters()},
            {"params": self.fusion.parameters()},
            {"params": self.classifier.parameters()},
        ]

        if self.use_prototypes:
            param_groups.insert(1, {"params": self.prototypes.parameters()})

        optimizer = torch.optim.SGD(
            param_groups,
            lr=self.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.wd,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.max_epochs,
            eta_min=0.0,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}





class CLMBREncoders(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        batch_processor=None,
        normalize: bool = False,
        freeze: bool = False,
        device: str = None,
    ):
        super().__init__()

        self.model = model
        self.batch_processor = batch_processor
        self.normalize = normalize
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device_)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        batch: Dict[str, Any],
        query_key: str = "query",
        history_key: str = "history",
    ) -> Dict[str, torch.Tensor]:

        q_batch = batch[query_key]
        h_batch = batch[history_key]

        q_vec = self._encode_packed_batch(q_batch)  # [B, H]

        h_vec_flat = self._encode_packed_batch(h_batch)  # [B*K, H]

        B, K = batch["history_valid_mask"].shape
        H = h_vec_flat.shape[-1]

        h_vec = h_vec_flat.reshape(B, K, H)

        return {
            "query_vec": q_vec,
            "hist_vec": h_vec,
        }

    def _encode_packed_batch(
        self,
        raw_batch: Mapping[str, Any],
    ) -> torch.Tensor:
#         if self.batch_processor is not None:
#             batch = self.batch_processor.collate([raw_batch])["batch"]
#             batch = femr.models.transformer.remove_first_dimension(batch)
#         else:
        batch = raw_batch

        transformer_batch = self._move_transformer_to_device(batch["transformer"])

        reprs = self.model.transformer(transformer_batch)

        patient_lengths = torch.as_tensor(
            transformer_batch["patient_lengths"],
            dtype=torch.long,
            device=reprs.device,
        )

        end_positions = torch.cumsum(patient_lengths, dim=0) - 1

        vec = reprs[end_positions]

        if self.normalize:
            vec = nn.functional.normalize(vec, p=2, dim=-1)

        return vec

    def _move_transformer_to_device(
        self,
        transformer_batch: Mapping[str, Any],
    ) -> Dict[str, Any]:

        out = {}

        for k, v in transformer_batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device_)
            else:
                out[k] = torch.as_tensor(v).to(self.device_)

        return out


class CLMBRRAPEvalModel(lt.LightningModule):
    def __init__(
        self,
        clmbr_model,
        batch_processor=None,
        hidden_size: int = 768,
        lr: float = 2e-5,
        wd: float = 0.001,
        max_epochs: int = 100,
        dropout: float = 0.1,
        freeze: bool = False,
        # --- prototype module ---
        num_prototypes: int = 256,
        query_temperature: float = 0.1,
        history_temperature: float = 0.025,
        normalize_prototypes: bool = True,
        softmax_threshold: float = 0.8,
        softmax_temperature: float = 1.0,
        sample_ent_lambda: float = 0.001,
        usage_ent_lambda: float = 0.001,
        use_prototypes: bool = False,
        # --- fusion module ---
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        fusion_ff_mult: int = 4,
        fusion_output_mode: str = "query",
        use_weights_as_gating: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["clmbr_model", "batch_processor"])

        self.hidden_size = hidden_size
        self.sample_ent_lambda = float(sample_ent_lambda)
        self.usage_ent_lambda = float(usage_ent_lambda)
        self.use_prototypes = use_prototypes

        self.encoders = CLMBREncoders(
            model=clmbr_model,
            batch_processor=batch_processor,
            normalize=False,
            freeze=freeze,
        )

        self.prototypes = PrototypeRetrievalModule(
            hidden_size=hidden_size,
            num_prototypes=num_prototypes,
            query_temperature=query_temperature,
            history_temperature=history_temperature,
            normalize_prototypes=normalize_prototypes,
            softmax_threshold=softmax_threshold,
            softmax_temperature=softmax_temperature,
        )

        self.fusion = FusionModule(
            hidden_size=hidden_size,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            ff_mult=fusion_ff_mult,
            dropout=dropout,
            use_weights_as_gating=use_weights_as_gating,
            output_mode=fusion_output_mode,
            return_seq=False,
        )

        self.classifier = nn.Linear(hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        self.train_auroc = BinaryAUROC()
        self.train_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

        self._train_preds, self._train_labels = [], []
        self._val_preds, self._val_labels = [], []
        self._test_preds, self._test_labels = [], []

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        enc_out = self.encoders(batch, query_key="query", history_key="history")

        proto_out = None
        attn_mask = None
        attn_weights = None
        hist_valid_mask = batch["history_valid_mask"]

        if self.use_prototypes:
            proto_out = self.prototypes(
                query_vec=enc_out["query_vec"],
                hist_vec=enc_out["hist_vec"],
                hist_valid_mask=hist_valid_mask,
            )
            attn_mask = proto_out["attn_mask"]
            attn_weights = proto_out["attn_weights"]

        fuse_out = self.fusion(
            query_vec=enc_out["query_vec"],
            hist_vec=enc_out["hist_vec"],
            attn_mask=attn_mask,
            attn_weights=attn_weights,
            hist_valid_mask=hist_valid_mask,
        )

        fused_vec = fuse_out["fused_vec"]
        logits = self.classifier(fused_vec).squeeze(-1)

        out = {
            "logits": logits,
            "fused_vec": fused_vec,
        }

        if proto_out is not None:
            out["proto"] = proto_out

        return out

    def shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        out = self.forward(batch)
        proto = out.get("proto", None)

        if proto is not None and "diag_q_ent" in proto:
            self.log(
                f"{stage}_sample_q_ent",
                proto["diag_q_ent"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_q_max",
                proto["diag_q_max"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_sample_h_ent",
                proto["diag_h_ent"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_h_max",
                proto["diag_h_max"],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        if proto is not None and "diag_mean_q_ent" in proto:
            self.log(
                f"{stage}_batch_q_ent",
                proto["diag_mean_q_ent"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_batch_h_ent",
                proto["diag_mean_h_ent"],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        logits = out["logits"]
        y = batch["label"].float().view(-1)

        task_loss = self.criterion(logits, y)

        sample_ent_reg = task_loss.new_zeros(())
        usage_ent_reg = task_loss.new_zeros(())

        if proto is not None and "diag_q_ent" in proto and self.sample_ent_lambda > 0.0:
            sample_ent_reg = proto["diag_q_ent"] + proto["diag_h_ent"]

        if proto is not None and "diag_mean_q_ent" in proto and self.usage_ent_lambda > 0.0:
            usage_ent_reg = -(proto["diag_mean_q_ent"] + proto["diag_mean_h_ent"])

        total_loss = (
            task_loss
            + self.sample_ent_lambda * sample_ent_reg
            + self.usage_ent_lambda * usage_ent_reg
        )

        self.log(
            f"{stage}_loss",
            task_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_total_loss",
            total_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        pos_score = torch.sigmoid(logits)

        if stage == "train":
            self._train_labels.append(y.detach())
            self._train_preds.append(pos_score.detach())
        elif stage == "val":
            self._val_labels.append(y.detach())
            self._val_preds.append(pos_score.detach())
        elif stage == "test":
            self._test_labels.append(y.detach())
            self._test_preds.append(pos_score.detach())

        if proto is not None and "attn_weights" in proto and proto["attn_weights"] is not None:
            with torch.no_grad():
                weights = proto["attn_weights"].float()
                mask = batch["history_valid_mask"].float()

                if not self.fusion.use_weights_as_gating:
                    hard_keep = (weights >= self.prototypes.softmax_threshold).float()
                    keep_rate = (hard_keep * mask).sum() / mask.sum().clamp_min(1.0)

                    self.log(
                        f"{stage}_keep_rate",
                        keep_rate,
                        prog_bar=False,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )

                else:
                    weights = weights * mask
                    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    weights_norm = weights / denom

                    weight_entropy = -(
                        weights_norm * weights_norm.clamp_min(1e-8).log()
                    ).sum(dim=1)

                    valid_counts = mask.sum(dim=1).clamp_min(1.0)
                    max_entropy = valid_counts.log().clamp_min(1e-8)
                    weight_entropy_norm = (weight_entropy / max_entropy).mean()

                    self.log(
                        f"{stage}_weight_entropy",
                        weight_entropy.mean(),
                        prog_bar=False,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )
                    self.log(
                        f"{stage}_weight_entropy_norm",
                        weight_entropy_norm,
                        prog_bar=False,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )

        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        if not self._train_preds:
            return

        y = torch.cat(self._train_labels)
        p = torch.cat(self._train_preds)

        self.log("train_auroc",self.val_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True,)
        self.log("train_auprc",self.val_auprc(p, y.long()),on_epoch=True, logger=True, prog_bar=True, sync_dist=True,)

        self._train_labels.clear()
        self._train_preds.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        y = torch.cat(self._val_labels)
        p = torch.cat(self._val_preds)

        self.log("val_auroc",self.val_auroc(p, y.long()), on_epoch=True, logger=True, prog_bar=True, sync_dist=True,)
        self.log("val_auprc",self.val_auprc(p, y.long()),on_epoch=True, logger=True, prog_bar=True, sync_dist=True,)

        self._val_labels.clear()
        self._val_preds.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        y = torch.cat(self._test_labels)
        p = torch.cat(self._test_preds)

        self.log("test_auroc", self.test_auroc(p, y.long()), on_epoch=True, logger=True)
        self.log("test_auprc", self.test_auprc(p, y.long()), on_epoch=True, logger=True)

        log_bootstrap_ci_text_percentile(module=self,
                                         y_true=y,
                                         y_score=p,
                                         prefix="test",
                                         num_iter=1000,
                                         alpha=0.05,
                                         ndigits=3,)

        self._test_labels.clear()
        self._test_preds.clear()

    def configure_optimizers(self):
        param_groups = [{"params": self.encoders.parameters()},
                        {"params": self.fusion.parameters()},
                        {"params": self.classifier.parameters()}]

        if self.use_prototypes:
            param_groups.insert(1, {"params": self.prototypes.parameters()})

        optimizer = torch.optim.SGD(param_groups,
                                    lr=self.lr,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=self.wd)

#         optimizer = torch.optim.AdamW(param_groups,
#                                     lr=self.lr,
# #                                     momentum=0.9,
# #                                     nesterov=True,
#                                     betas=(0.9,0.95),
#                                     weight_decay=self.wd)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.max_epochs,
                                                               eta_min=0.0)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}