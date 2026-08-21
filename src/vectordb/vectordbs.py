
import os
import json
import torch
import faiss
import femr
import numpy as np
import polars as pl

import torch.nn as nn
from tqdm import tqdm
from datasets import load_from_disk
from collections import defaultdict
from ..models.models import EHREmbeddings
from ..data.datasets import SequencesGenerator
from typing import Any, List, Dict, Union, Optional, Mapping 
from ..models.utils import get_config_and_model_cls
from .utils import build_index, to_list, build_faiss_index
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from ..data.utils import pack_clmbr_chunks







class CLMBREmbedder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        batch_processor=None,
        normalize: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model = model
        self.batch_processor = batch_processor
        self.normalize = normalize
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval().to(self.device_)

    @torch.no_grad()
    def encode(self, raw_batch: Mapping[str, Any]) -> torch.Tensor:
        if self.batch_processor is not None:
            batch = self.batch_processor.collate([raw_batch])["batch"]
            batch = femr.models.transformer.remove_first_dimension(batch)
        else:
            batch = raw_batch

        transformer_batch = self._move_transformer_to_device(batch["transformer"])

        reprs = self.model.transformer(transformer_batch)

        patient_lengths = torch.as_tensor(
            transformer_batch["patient_lengths"],
            dtype=torch.long,
            device=reprs.device,
        )

        end_positions = torch.cumsum(patient_lengths, dim=0) - 1
        embs = reprs[end_positions]

        if self.normalize:
            embs = nn.functional.normalize(embs, p=2, dim=-1)

        return embs

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





class EHREmbedder(nn.Module):
    def __init__(
        self,
        config,
        backbone,
        ckpt_path: Optional[str] = None,
        dropout: float = 0.1,
        pooling: str = "cls",
        normalize: bool = False,
        use_numeric: bool = False, 
        use_time: bool = False,
        use_type: bool = True,
        use_visit: bool = True,
        use_stage: bool = True,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = config
        self.pooling = pooling
        self.normalize = normalize
        self.dtype_ = dtype
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone(config)
        self.use_time = use_time
        self.use_numeric = use_numeric
        self.use_type = use_type
        self.use_visit = use_visit
        self.use_stage = use_stage
            
        rope_model_types = {"modernbert", "roformer"}
        model_type = getattr(config, "model_type", "").lower()
        is_rope = model_type in rope_model_types
        
        self.ehr_embeddings = EHREmbeddings(
            vocab_size=config.vocab_size,
            embedding_size=getattr(config, "hidden_size", getattr(config, "embedding_size")),
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            visit_vocab_size=config.visit_vocab_size,
            stage_vocab_size=config.stage_vocab_size,
            dropout=dropout,
            use_position_embeddings=not is_rope,
            max_position_embeddings=(getattr(config, "max_position_embeddings", 0) if not is_rope else 0),
            use_time=self.use_time,
            time_in_features=1,
            time_out_features=16,
            use_numeric=self.use_numeric,
            use_type=self.use_type,
            use_visit=self.use_visit,
            use_stage=self.use_stage,
        )

        if ckpt_path:
            self.load_pretrained_weights(ckpt_path)

        self.eval().to(self.device_, dtype=self.dtype_)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        type_ids: torch.Tensor = None,
        visit_ids: torch.Tensor = None,
        stage_ids: torch.Tensor = None,
        time_feats: Optional[torch.Tensor] = None,
        numeric_values: Optional[torch.Tensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_time and time_feats is None:
            raise ValueError("time_feats must be provided when use_time=True")

        if self.use_numeric and (numeric_values is None or numeric_mask is None):
            raise ValueError("numeric_values and numeric_mask must be provided when use_numeric=True")

        if self.use_type and type_ids is None:
            raise ValueError("Type ids must be provided when use_type=True")

        if self.use_visit and visit_ids is None:
            raise ValueError("visit ids must be provided when use_visit=True")

        if self.use_stage and stage_ids is None:
            raise ValueError("stage ids must be provided when use_stage=True")
        
        inputs_embeds = self.ehr_embeddings.encode(
            input_ids=input_ids.to(self.device_),
            type_ids=type_ids.to(self.device_) if (self.use_type and type_ids is not None) else None,
            visit_ids=visit_ids.to(self.device_) if (self.use_visit and visit_ids is not None) else None,
            stage_ids=stage_ids.to(self.device_) if (self.use_stage and stage_ids is not None) else None,
            time_feats=time_feats.to(self.device_) if (self.use_time and time_feats is not None) else None,
            numeric_values=numeric_values.to(self.device_) if (self.use_numeric and numeric_values is not None) else None,
            numeric_mask=numeric_mask.to(self.device_) if (self.use_numeric and numeric_mask is not None) else None,
        ).to(self.dtype_)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(self.device_),
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  

        if self.pooling == "cls":
            vec = last_hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(device=last_hidden.device,dtype=last_hidden.dtype) 
            vec = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        else:
            raise ValueError(f"Unsupported pooling='{self.pooling}' (use 'cls' or 'mean')")

        if self.normalize:
            vec = nn.functional.normalize(vec, p=2, dim=-1)
        return vec

    def load_pretrained_weights(self, ckpt_path: str) -> None:
        sd_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd_obj["state_dict"] if isinstance(sd_obj, dict) and "state_dict" in sd_obj else sd_obj

        mt = getattr(self.config, "model_type", "").lower()
        prefix_map = {
            "bert":       "backbone.bert.",
            "roberta":    "backbone.roberta.",
            "longformer": "backbone.longformer.",
            "modernbert": "backbone.model.",
            "roformer":   "backbone.roformer.",
            "big_bird":   "backbone.bert.",
            "mamba":      "backbone.backbone.",
            "mamba2":     "backbone.backbone."}
        backbone_prefix = prefix_map.get(mt, None)

        DROP_PREFIXES = ["backbone.cls.", "top_1_train.", "top_1_val.", "backbone.lm_head.", 
                         "classifier.", "cls.", "lm_head.", "score."]
        remapped = {}
        for k, v in sd.items():
            if any(k.startswith(dp) for dp in DROP_PREFIXES):
                continue

            if k.startswith("ehr_embeddings."):
                new_k = k
            elif backbone_prefix and k.startswith(backbone_prefix):
                new_k = "backbone." + k[len(backbone_prefix):]
            elif k.startswith("backbone."):
                new_k = k
            else:
                new_k = k

            remapped[new_k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print("Embedder weights loaded.")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)


class EmbedCollator:
    def __init__(self) -> None:
        self.embed_keys = ("input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids")

    def __call__(self, batch: List[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        chunks = self._flatten(batch)
        return self._stack(chunks)

    def _flatten(self, batch) -> List[Dict]:
        out = []
        for item in batch:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                out.extend(item)
            else:
                raise TypeError(f"Unexpected item type: {type(item)}")
        if not out:
            raise ValueError("Empty batch after normalization.")
        return out

    def _stack(self, chunks: List[Dict]) -> Dict[str, torch.Tensor]:
        out = {}
        for k in self.embed_keys:
            seq_list = []
            for c in chunks:
                v = c.get(k, None)
                if isinstance(v, list):
                    v = [0 if x is None else x for x in v]
                elif v is None:
                    v = 0
                seq_list.append(torch.as_tensor(v))
            out[k] = torch.stack(seq_list, 0)
        return out
    


class ChromaEHREmbeddingFunction:
    def __init__(self, embedder, collate_fn, batch_size: Optional[int] = None):
        self.embedder = embedder
        self.collate_fn = collate_fn
        self.batch_size = batch_size 

    def _decode_docs(self, docs: Documents):
        out: List[Union[dict, list]] = []
        for item in docs:
            if isinstance(item, str):
                out.append(json.loads(item))
            elif isinstance(item, (list, tuple)):
                out.append([json.loads(x) if isinstance(x, str) else x for x in item])
            else:
                out.append(item)
        return out

    def _slice_batch(self, batch_tensors: Dict[str, torch.Tensor], sl: slice) -> Dict[str, torch.Tensor]:
        # Required
        keys = ["input_ids", "attention_mask", "type_ids", "visit_ids", "stage_ids"]
        # Optional (only pass if present)
        for opt in ["time_diff", "numeric_values", "numeric_mask"]:
            if opt in batch_tensors:
                keys.append(opt)
        return {k: batch_tensors[k][sl] for k in keys}
    
    def name(self) -> str:
        return self.embedder.config.model_type
    
    def __call__(self, input: Documents) -> Embeddings:
        # 1) Decode JSON if needed
        docs = self._decode_docs(input)
        batch_inputs = self.collate_fn(docs)
        device = getattr(self.embedder, "device_", "cpu")

        req_long = ["input_ids", "attention_mask", "type_ids", "visit_ids", "stage_ids"]
        
        for k in req_long:
            if k not in batch_inputs:
                raise KeyError(f"Missing required key '{k}' in collated batch.")
            batch_inputs[k] = batch_inputs[k].to(device=device, dtype=torch.long)

        if "time_diff" in batch_inputs:
            batch_inputs["time_diff"] = batch_inputs["time_diff"].to(device=device)
        if "numeric_values" in batch_inputs:
            batch_inputs["numeric_values"] = batch_inputs["numeric_values"].to(device=device)
        if "numeric_mask" in batch_inputs:
            batch_inputs["numeric_mask"] = batch_inputs["numeric_mask"].to(device=device)

        B = batch_inputs["input_ids"].size(0)
        bs = self.batch_size or B

        embs_out: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, B, bs):
                end = min(start + bs, B)
                sub = self._slice_batch(batch_inputs, slice(start, end))
                v = self.embedder.encode(
                    input_ids=sub["input_ids"],
                    attention_mask=sub["attention_mask"],
                    type_ids=sub["type_ids"],
                    visit_ids=sub["visit_ids"],
                    stage_ids=sub["stage_ids"],
#                     time_feats=sub.get("time_diff", None), uncomment if use time is true and want it to be embedded
#                     numeric_values=sub.get("numeric_values", None), Same above
#                     numeric_mask=sub.get("numeric_mask", None), same above
                )
                embs_out.append(v.detach().cpu())

        return torch.cat(embs_out, dim=0).tolist()
    


class VectorDBUploader:
    def __init__(
        self,
        collection,
        seq_gen,
        main_window: str,
        seq_length: int,
        limits: dict,
        data_idx_path: str,
        data_path: str,
        use_time: bool = False, # set as true for values storage only
        use_numeric: bool = False, # set as true for values storage only
        add_cls_per_chunk: bool = True,
        needed_cols: list = None,
    ):
        self.collection = collection
        self.seq_gen = seq_gen
        self.main_window = main_window
        self.seq_length = seq_length
        self.limits = limits
        self.use_time = use_time 
        self.use_numeric = use_numeric
        self.add_cls_per_chunk = add_cls_per_chunk

        if needed_cols is None:
            needed_cols = ["subject_id", "input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids"]
            
            if self.use_time :
                needed_cols.append("time_diff")
            if self.use_numeric:
                needed_cols += ["numeric_values", "numeric_mask"]
                
        self.needed_cols = needed_cols
        
        
        self.data_idx = pl.read_parquet(data_idx_path)
        sub_ids = set(self.data_idx.get_column("subject_id").to_list())

        hf_dataset = load_from_disk(data_path)
        hf_dataset = hf_dataset.filter(
            lambda sids: [sid in sub_ids for sid in sids],
            batched=True,
            input_columns="subject_id")

        self.hf_dataset = (hf_dataset
                           .flatten_indices()
                           .select_columns(self.needed_cols)
                           .with_format("numpy", columns=self.needed_cols, 
                                        output_all_columns=False))

        self.index = self._build_index()

    def _build_index(self):
        sids = self.hf_dataset["subject_id"]
        index = defaultdict(list)
        for i, sid in enumerate(sids):
            index[sid].append(i)
        return index

    def _to_list(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return v
        return v

#     def _ensure_optional_fields(self, chunk: dict):
#         L = len(chunk["input_ids"])
#         if self.use_time:
#             if "time_diff" not in chunk or chunk["time_diff"] is None:
#                 chunk["time_diff"] = [0.0] * L
#             else:
#                 chunk["time_diff"] = [0.0 if x is None else float(x) for x in chunk["time_diff"]]
#         if self.use_numeric:
#             if "numeric_values" not in chunk or chunk["numeric_values"] is None:
#                 chunk["numeric_values"] = [0.0] * L
#             else:
#                 chunk["numeric_values"] = [0.0 if x is None else float(x) for x in chunk["numeric_values"]]
#             if "numeric_mask" not in chunk or chunk["numeric_mask"] is None:
#                 chunk["numeric_mask"] = [0] * L
#             else:
#                 chunk["numeric_mask"] = [0 if x is None else int(x) for x in chunk["numeric_mask"]]

#         return chunk

    def upsert_chunks(self):
        for row_idx in tqdm(range(len(self.data_idx))):
            stay = self.data_idx[row_idx]
            split = stay["split"][0]
            
            subject_id = stay["subject_id"][0]
            
            start_key_or_int = self.limits[self.main_window][self.seq_length][0]
            end_key = self.limits[self.main_window][self.seq_length][1]
            
            start = stay[start_key_or_int][0] if isinstance(start_key_or_int, str) else int(start_key_or_int)
            end = stay[end_key][0] if isinstance(end_key, str) else int(end_key)
            
            timeline_encoded = self.hf_dataset.select(self.index[subject_id])[0]
            
            history_window = {k: self._to_list(v) for k, v in timeline_encoded.items()}
            history_window = {k: (v[start:end] if isinstance(v, list) else v) for k, v in history_window.items()}
            
            chunks = self.seq_gen.get_overlapped_chunks(history_window,
                                                        add_cls_per_chunk=self.add_cls_per_chunk)
            docs, ids, metas = [], [], []
            
            for ci, ch in enumerate(chunks):
#                 ch = self._ensure_optional_fields(ch)
                doc_dict = {"input_ids": ch["input_ids"],
                            "attention_mask": ch["attention_mask"],
                            "visit_ids": ch["visit_ids"],
                            "stage_ids": ch["stage_ids"],
                            "type_ids": ch["type_ids"]}
                docs.append(json.dumps(doc_dict, separators=(",", ":")))
                ids.append(f"{subject_id}__{row_idx:06d}__{ci:04d}")
                meta = {"subject_id": int(subject_id),
                        "stay_row": int(row_idx),
                        "chunk_idx": int(ci),
                        "split": str(split),
                        "window": str(self.main_window),
                        "seq_length": int(self.seq_length),}

                if self.use_time:
                    meta["time_diff"] = json.dumps(ch["time_diff"], separators=(",", ":"))
                if self.use_numeric:
                    meta["numeric_values"] = json.dumps(ch["numeric_values"], separators=(",", ":"))
                    meta["numeric_mask"] = json.dumps(ch["numeric_mask"], separators=(",", ":"))

                metas.append(meta)

            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            del timeline_encoded, history_window, chunks, docs, ids, metas





def build_indices(data_idx_path:str,
                  hf_dataset_path: str,
                  tokenizer_path: str,
                  save_path:str,
                  main_window_q: str,
                  main_window_h: str,
                  limits_dict: dict,
                  ckpt_path: str,
                  embedder_model: str = 'roformer',
                  chunking_strategy:str='overlap',
                  seq_length_q: int =1024,
                  overlap_q: int=0,
                  seq_length_h: int =256,
                  overlap_h: int= 0,
                  window_hours: float= 6.0,
                  use_time: bool = False,
                  use_numeric: bool = False,
                  use_type: bool = True,
                  use_visit: bool = True,
                  use_stage: bool = True,
                 )-> None:
    
    assert chunking_strategy in ['overlap','time','visit','care_stage']

    seq_gen_q = SequencesGenerator(tokenizer_path=tokenizer_path,
                                   chunk_length=seq_length_q,
                                   overlap=overlap_q)
    
    seq_gen_h = SequencesGenerator(tokenizer_path=tokenizer_path,
                                   chunk_length=seq_length_h,
                                   overlap=overlap_h)
    
    
    data_idx = pl.read_parquet(data_idx_path)
    hf_dataset= load_from_disk(hf_dataset_path)
    
    ConfigClassRet, ModelClassRet = get_config_and_model_cls(model_type=embedder_model, mode='eval', variant=None)

    cfg_ret = ConfigClassRet(vocab_size=seq_gen_q.tokenizer.vocab_size,
                             hidden_size=768,
                             num_hidden_layers=12,
                             num_attention_heads=12,
                             intermediate_size=3072,
                             max_position_embeddings=1536,
                             pad_token_id=0,
                             type_vocab_size= 43,
                             visit_vocab_size= 990,
                             stage_vocab_size= 5)

    embedder = EHREmbedder(
        config=cfg_ret,
        backbone=ModelClassRet,
        ckpt_path=ckpt_path,
        pooling='mean',
        use_time=use_time,
        use_numeric=use_numeric,
        use_type=use_type,
        use_visit=use_visit,
        use_stage=use_stage,
        )
    
    
    if chunking_strategy == 'overlap':
        needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids']
    elif chunking_strategy == 'time':
        needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids','time_stamp']
    elif chunking_strategy == 'visit':
        needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids',]
    elif chunking_strategy == 'care_stage':
        needed_cols = ['subject_id','input_ids','attention_mask','visit_ids','stage_ids','type_ids',]

    if use_time:
        needed_cols.append('time_diff')
    if use_numeric:
        needed_cols.append('numeric_values')
        needed_cols.append('numeric_mask')
    
    sub_ids = set(data_idx.get_column("subject_id").to_list())
    hf_dataset = hf_dataset.filter(lambda sids: [sid in sub_ids for sid in sids],
                                   batched=True,
                                   input_columns="subject_id")

    hf_dataset = (hf_dataset
                       .flatten_indices()
                       .select_columns(needed_cols)
                       .with_format("numpy", columns=needed_cols, 
                                    output_all_columns=False))

    index =  build_index(hf_dataset)
    
    for i in tqdm(range(len(data_idx))):
        row_idx = i
        stay = data_idx[row_idx]

        subject_id = stay['subject_id'][0]
        stay_id = stay['icustay_id'][0]

        timeline_encoded = hf_dataset.select(index[subject_id])[0]

        start_limit_q = limits_dict[main_window_q][seq_length_q][0]
        end_limit_q = limits_dict[main_window_q][seq_length_q][1]
        start_q = stay[start_limit_q][0]
        end_q = stay[end_limit_q][0]


        query = {k: to_list(v) for k, v in timeline_encoded.items()}
        query = {k: (v[start_q:end_q] if isinstance(v, list) else v) for k, v in query.items()}
        query = seq_gen_q.get_overlapped_chunks(timeline=query,add_cls_per_chunk=True)[0]
        # query = {k: query[k] for k in ["input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids"] if k in query}
        query = {k: torch.tensor(v) for k, v in query.items()}


        start_limit_h = limits_dict[main_window_h][seq_length_q][0]
        end_limit_h = limits_dict[main_window_h][seq_length_q][1]
        start_h = stay[start_limit_h][0] if isinstance(start_limit_h, str) else int(start_limit_h)
        end_h = stay[end_limit_h][0] if isinstance(end_limit_h, str) else int(end_limit_h)

        
        history = {k: to_list(v) for k, v in timeline_encoded.items()} 
        history = {k: (v[start_h:end_h] if isinstance(v, list) else v) for k, v in history.items()}
        
        if chunking_strategy == 'overlap':
            history = seq_gen_h.get_overlapped_chunks(timeline=history,
                                                      add_cls_per_chunk=True)
        elif chunking_strategy == 'time':
            history = seq_gen_h.get_time_based_chunks(timeline=history,
                                                     window_hours=window_hours,
                                                     keep_prefix_tokens=True,
                                                     anchor_from_first_valid_time=True)
        elif chunking_strategy == 'visit':
            history = seq_gen_h.get_visit_level_chunks(timeline=history,
                                                      keep_prefix_tokens=True)
        elif chunking_strategy == 'care_stage':
            history = seq_gen_h.get_care_stage_level_chunks(timeline=history,
                                                           keep_prefix_tokens=True)
            
        # history = [{k: chunk[k] for k in ["input_ids", "attention_mask", "visit_ids", "stage_ids", "type_ids"] 
        #             if k in chunk} for chunk in history]
        history = [{k: torch.tensor(v) for k, v in chunk.items()} for chunk in history]
        history = {k: torch.stack([c[k] for c in history], dim=0) for k in history[0].keys()}

        q_emb = embedder.encode(
            input_ids=query['input_ids'].unsqueeze(0),
            attention_mask=query['attention_mask'].unsqueeze(0),
            type_ids=query['type_ids'].unsqueeze(0) if use_type else None,
            visit_ids=query['visit_ids'].unsqueeze(0) if use_visit else None,
            stage_ids=query['stage_ids'].unsqueeze(0) if use_stage else None,
            time_feats=query['time_diff'].unsqueeze(0) if use_time else None,
            numeric_values=query['numeric_values'].unsqueeze(0) if use_numeric else None,
            numeric_mask=query['numeric_mask'].unsqueeze(0) if use_numeric else None,
        )

        ch_emb = embedder.encode(
            input_ids=history['input_ids'],
            attention_mask=history['attention_mask'],
            type_ids=history['type_ids'] if use_type else None,
            visit_ids=history['visit_ids'] if use_visit else None,
            stage_ids=history['stage_ids'] if use_stage else None,
            time_feats=history['time_diff'] if use_time else None,
            numeric_values=history['numeric_values'] if use_numeric else None,
            numeric_mask=history['numeric_mask'] if use_numeric else None,
        )

        ch_emb = torch.cat((ch_emb,q_emb),dim=0)

        idx, _ = build_faiss_index(ch_embs=ch_emb, metric='cosine')
        faiss.write_index(idx, os.path.join(save_path,f"{stay_id}.faiss"))






def build_clmbr_indices(
    data_idx_path: str,
    arrow_dataset_path: str,
    task: str,
    save_path: str,
    model,
    batch_processor,
    query_length: int = 512,
    history_chunk_length: int = 256,
    history_overlap: int = 0,
    chunking_strategy: str = "overlap",
    window_hours: float = 6.0,
):

    assert chunking_strategy in ["overlap", "time"]

#     os.makedirs(save_path, exist_ok=True)

    arrow_ds = load_from_disk(arrow_dataset_path)
    data_idx = pl.read_parquet(data_idx_path)
    data_idx = data_idx.filter(pl.col('task') == task)

    seq_gen = SequencesGenerator(
        tokenizer_path=None,
        chunk_length=history_chunk_length,
        overlap=history_overlap,
        dataset_name="ehrshot")

    embedder = CLMBREmbedder(model=model, batch_processor=batch_processor, normalize=False)

    for row_i in tqdm(range(len(data_idx))):
        row = data_idx[row_i]
        subject_id = int(row["subject_id"][0])
        example_id = int(row["example_id"][0])
        arrow_row_idx = int(row["arrow_row_idx"][0])
        prediction_idx = int(row["prediction_idx"][0])
        history_min_idx = int(row["history_min_idx"][0])
        sample = arrow_ds[arrow_row_idx]
        tr = sample["transformer"]

        timeline = {
            "tokens": to_list(tr["tokens"]),
            "valid_tokens": to_list(tr["valid_tokens"]),
            "ages": to_list(tr["ages"]),
            "normalized_ages": to_list(tr["normalized_ages"]),
            "timestamps": to_list(tr["timestamps"]),
            "patient_lengths": to_list(tr["patient_lengths"]),
            "label_indices": to_list(tr["label_indices"])}

        query_start = max(history_min_idx, prediction_idx - query_length)
        query_end = prediction_idx

        query = {k: v[query_start:query_end] for k, v in timeline.items() if isinstance(v, list)}

        query["patient_lengths"] = [len(query["tokens"])]
        query["label_indices"] = []

        query_batch = pack_clmbr_chunks([query])
        q_emb = embedder.encode(query_batch)

        history_start = history_min_idx
        history_end = query_start

        history = {k: v[history_start:history_end] for k, v in timeline.items() if isinstance(v, list)}

        history_chunks = []

        if len(history["tokens"]) > 0:
            if chunking_strategy == "overlap":
                history_chunks = seq_gen.get_overlapped_chunks(timeline=history, add_cls_per_chunk=False)

            elif chunking_strategy == "time":
                history_chunks = seq_gen.get_time_based_chunks(timeline=history, window_hours=window_hours)

        if len(history_chunks) > 0:
            history_batch = pack_clmbr_chunks(history_chunks)
            h_emb = embedder.encode(history_batch)
            all_embs = torch.cat([h_emb, q_emb], dim=0)

        else:
            all_embs = q_emb

        index, _ = build_faiss_index(ch_embs=all_embs,metric="cosine")
        faiss_name = f"{subject_id}_{example_id}.faiss"
        faiss.write_index(index,os.path.join(save_path, faiss_name))