
import torch
import random
import numpy as np
import polars as pl
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Tuple


#########################################################
# DescEmb model
#########################################################
class DescEmbDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_idx_path: str,
        task: str = "y_mort",
        main_window: str = "within48_descemb",  
        split: str = "train",
        max_word_len: int = 32,                 
        max_events: int = None,                 
    ) -> None:

        self.task = task
        self.main_window = main_window
        self.max_word_len = max_word_len
        self.max_events = max_events
        

        self.data_idx = pl.scan_parquet(data_idx_path).collect()
        self.data_idx = self.data_idx.filter(pl.col("split") == split)
        self.data_idx = self.data_idx.filter(~pl.col("subject_id").is_in([15409850,16816440,18757959]) )


        self.hf_dataset = load_from_disk(dataset_path)

        subj_ids = self.hf_dataset["subject_id"]
        icu_ids = self.hf_dataset["icustay_id"]
        self._hf_index = {
            (int(s), int(i)): idx for idx, (s, i) in enumerate(zip(subj_ids, icu_ids))
        }


        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

    def __len__(self) -> int:
        return len(self.data_idx)

    def __getitem__(self, idx: int):
        row = self.data_idx.row(idx, named=True)
        sid = int(row["subject_id"])
        hadm = int(row["hadm_id"])
        icu = int(row["icustay_id"])
        y = row[self.task]

        hf_idx = self._hf_index.get((sid, icu))
        if hf_idx is None:
            return None

        ex = self.hf_dataset[hf_idx]          
        events = ex[self.main_window] or []
        events = [e for e in events if isinstance(e, str) and e.strip()]
        if self.max_events is not None:
            events = events[: self.max_events]
        if len(events) == 0:
            return None

        enc = self.tokenizer(
            events,
            padding="max_length",
            truncation=True,
            max_length=self.max_word_len,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "subject_id": sid,
            "hadm_id": hadm,
            "icustay_id": icu,
            "input_ids": enc["input_ids"].long(),
            "attention_mask": enc["attention_mask"].long(),
            "seq_len": torch.tensor(len(events), dtype=torch.long),
            "label": torch.tensor(float(y), dtype=torch.float32),
        }
    


class DescEmbCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):


        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}

        lengths = [int(b["seq_len"]) for b in batch]
        max_S = max(lengths)
        W = batch[0]["input_ids"].shape[1]
        B = len(batch)

        input_ids = torch.full((B, max_S, W), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_S, W), dtype=torch.long)
        seq_len = torch.tensor(lengths, dtype=torch.long)
        labels = torch.stack([b["label"] for b in batch])

        for i, b in enumerate(batch):
            S_i = b["input_ids"].shape[0]
            input_ids[i, :S_i] = b["input_ids"]
            attention_mask[i, :S_i] = b["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_len": seq_len,
            "label": labels,

            "subject_id": torch.tensor(
                [b["subject_id"] for b in batch],
                dtype=torch.long,
            ),

            "hadm_id": torch.tensor(
                [b["hadm_id"] for b in batch],
                dtype=torch.long,
            ),

            "icustay_id": torch.tensor(
                [b["icustay_id"] for b in batch],
                dtype=torch.long,
            ),
        }
    

#########################################################
# GenHPF model
#########################################################

class HierarchicalGenHPFDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_idx_path: str,
        seq_field: str,
        label_field: Optional[str] = None,
        split: str = "train",
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_events: int = 511,
        max_tokens: int = 128,
    ) -> None:
        self.hf_dataset = self.hf_dataset = load_from_disk(dataset_path)
        self.seq_field = seq_field
        self.label_field = label_field
        self.max_events = max_events
        self.max_tokens = max_tokens

        df = pl.scan_parquet(data_idx_path).collect()
        self.data_idx = df.filter(pl.col("split") == split).to_pandas()

        subj = self.hf_dataset["subject_id"]
        stay = self.hf_dataset["icustay_id"]
        key_to_hf_idx: Dict[Tuple[int, int], int] = {}
        for i, (s, h) in enumerate(zip(subj, stay)):
            key_to_hf_idx[(int(s), int(h))] = i
        self.key_to_hf_idx = key_to_hf_idx


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.data_idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data_idx.iloc[idx]
        subject_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        icustay_id = int(row["icustay_id"])

        key = (subject_id, icustay_id)
        if key not in self.key_to_hf_idx:
            return {"input_ids": None, "label": None}

        hf_idx = self.key_to_hf_idx[key]
        hf_row = self.hf_dataset[hf_idx]

        events: List[str] = hf_row[self.seq_field]

        if len(events) > self.max_events:
            events = events[: self.max_events]


        if len(events) == 0:
            return {"input_ids": None, "label": None}

        enc = self.tokenizer(
            events,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].long() 

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "icustay_id": icustay_id,
        }

        if self.label_field is not None:
            y = row[self.label_field]
            out["label"] = torch.tensor(float(y), dtype=torch.float32)

        return out
    

class GenHPFEvalCollator:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch = [b for b in batch if b["input_ids"] is not None]
        if len(batch) == 0:
            return {}

        input_ids_list = [b["input_ids"] for b in batch]  
        sizes = [x.size(0) for x in input_ids_list]
        B = len(input_ids_list)
        S_max = max(sizes)
        W = input_ids_list[0].size(1)


        collated_input_ids = input_ids_list[0].new_full(
            (B, S_max, W), fill_value=self.pad_token_id
        ).long()


        padding_mask = torch.ones(B, S_max, dtype=torch.bool)

        for i, (ids, S_i) in enumerate(zip(input_ids_list, sizes)):
            collated_input_ids[i, :S_i, :] = ids
            padding_mask[i, :S_i] = False

        out: Dict[str, Any] = {
            "input_ids": collated_input_ids,
            "padding_mask": padding_mask,
        }

        if "label" in batch[0] and batch[0]["label"] is not None:
            labels = torch.stack([b["label"] for b in batch])  # (B,)
            out["label"] = labels

        out["subject_id"] = torch.tensor(
            [b["subject_id"] for b in batch],
            dtype=torch.long,
        )

        out["hadm_id"] = torch.tensor(
            [b["hadm_id"] for b in batch],
            dtype=torch.long,
        )

        out["icustay_id"] = torch.tensor(
            [b["icustay_id"] for b in batch],
            dtype=torch.long,
        )

        return out
    
class GenHPFSimCLRDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_idx_path: str,
        split: str = "train",
        seq_field: str = "within_stay_remed",
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_events: int = 511,
        max_tokens: int = 128,
        min_events: int = 2,
        seed: int = 0,
    ):
        self.seq_field = seq_field
        self.max_events = max_events
        self.max_tokens = max_tokens
        self.min_events = min_events
        self.rng = random.Random(seed)


        df = pl.scan_parquet(data_idx_path).collect()
        df = df.filter(pl.col("split") == split)
        df = df.filter(~pl.col("subject_id").is_in([15409850, 16816440, 18757959]))
        self.data_idx = df.to_pandas()


        hf = load_from_disk(dataset_path)
        keep = [i for i, s in enumerate(hf["subject_id"]) if int(s) not in [15409850, 16816440, 18757959]]
        hf = hf.select(keep)
        self.hf_dataset = hf

        self.key_to_hf_idx: Dict[Tuple[int, int], int] = {
            (int(s), int(h)): i
            for i, (s, h) in enumerate(zip(hf["subject_id"], hf["icustay_id"]))
        }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.data_idx)

    def _consecutive_slice(self, events: List[str]) -> List[str]:
        n = len(events)

        if n <= self.max_events:
            return events

        start = self.rng.randint(0, n - self.max_events)
        return events[start : start + self.max_events]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data_idx.iloc[idx]
        key = (int(row["subject_id"]), int(row["icustay_id"]))

        hf_idx = self.key_to_hf_idx.get(key, None)
        if hf_idx is None:
            return {"input_ids": None}

        events = self.hf_dataset[hf_idx].get(self.seq_field, None)
        if not events:
            return {"input_ids": None}

        events = [e for e in events if isinstance(e, str) and e.strip()]
        if len(events) < self.min_events:
            return {"input_ids": None}

        events = self._consecutive_slice(events)

        enc = self.tokenizer(
            events,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"].long()}




class GenHPFSimCLRCollator:
    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        cls_token_id: int,          
        mask_prob: float = 0.15,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.mask_prob = mask_prob

    def __call__(self, batch):
        batch = [b for b in batch if b["input_ids"] is not None]
        if len(batch) == 0:
            return {}

        v1s, v2s = [], []

        for b in batch:
            ev = b["input_ids"]  
            S = ev.size(0)
            if S <= 1:
                v1, v2 = ev, ev
            else:
                mid = S // 2
                v1, v2 = ev[:mid, :], ev[mid:, :]
            v1s.append(v1)
            v2s.append(v2)

        views = v1s + v2s  

        sizes = [v.size(0) for v in views]
        B2 = len(views)
        S_max = max(sizes)
        W = views[0].size(1)

        collated_input_ids = views[0].new_full(
            (B2, S_max, W), fill_value=self.pad_token_id
        ).long()

        padding_mask = torch.ones(B2, S_max, dtype=torch.bool) 

        for i, (v, S_i) in enumerate(zip(views, sizes)):
            collated_input_ids[i, :S_i, :] = v
            padding_mask[i, :S_i] = False

        b_idx, s_idx = torch.where(padding_mask)     # both are 1d, same length
        collated_input_ids[b_idx, s_idx, 0] = self.cls_token_id
        ids = collated_input_ids
        rand = torch.rand_like(ids, dtype=torch.float32)
        mask = (rand < self.mask_prob) & (ids != self.pad_token_id)
        ids[mask] = self.mask_token_id

        return {"input_ids": ids, "padding_mask": padding_mask}
    

#########################################################
# REMed model
#########################################################


class REMedGenHPFPoolDataset(Dataset):

    def __init__(
        self,
        hf_path: str,                 
        data_idx_path: str,
        seq_field: str,               
        time_field: str,              
        time_diff_field: str,         
        label_field: Optional[str] = None,
        split: str = "train",
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        seq_len: int = 512,
        max_tokens: int = 128,
        random_sample_train: bool = True,
        deterministic_eval: bool = True,
        seed: int = 2020,
    ):
        full_ds = load_from_disk(hf_path)


        self.hf_dataset = full_ds.select_columns([
            "subject_id",
            "icustay_id",
            seq_field,
            time_field,
            time_diff_field,
        ])
        self.seq_field = seq_field
        self.time_field = time_field
        self.time_diff_field = time_diff_field
        self.label_field = label_field
        self.seq_len = int(seq_len)
        self.max_tokens = int(max_tokens)
        self.random_sample_train = random_sample_train
        self.deterministic_eval = deterministic_eval
        self.split = split


        subj = self.hf_dataset["subject_id"]
        stay = self.hf_dataset["icustay_id"]
        self.key_to_hf_idx: Dict[Tuple[int, int], int] = {
            (int(s), int(h)): i for i, (s, h) in enumerate(zip(subj, stay))
        }

        df = pl.scan_parquet(data_idx_path).collect()
        self.data_idx = df.filter(pl.col("split") == split).to_pandas()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        self.base_seed = int(seed)

    def __len__(self) -> int:
        return len(self.data_idx)

    def _rng_for(self, subject_id: int, icustay_id: int) -> np.random.Generator:
        
        mix = (self.base_seed * 1_000_003) ^ (subject_id * 1009) ^ (icustay_id * 9176)
        mix = mix % (2**32 - 1)
        return np.random.default_rng(mix)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data_idx.iloc[idx]
        subject_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        icustay_id = int(row["icustay_id"])

        key = (subject_id, icustay_id)
        if key not in self.key_to_hf_idx:
            return {"input_ids": None, "label": None}

        hf_row = self.hf_dataset[self.key_to_hf_idx[key]]

        events: List[str] = list(hf_row[self.seq_field])
        times_ts = list(hf_row[self.time_field])
        time_diff = list(hf_row[self.time_diff_field])


        L = len(events)
        if L == 0 or L != len(times_ts) or L != len(time_diff):
            return {"input_ids": None, "label": None}


        if self.split == "train" and self.random_sample_train:
            rng = self._rng_for(subject_id, icustay_id)
            if L >= self.seq_len:

                idxs = rng.choice(L, size=self.seq_len, replace=False)
            else:

                idxs = rng.choice(L, size=self.seq_len, replace=True)

            idxs = np.sort(idxs)
        else:
            # eval: deterministic
            if self.deterministic_eval:
                idxs = np.arange(max(0, L - self.seq_len), L)
            else:
                rng = self._rng_for(subject_id, icustay_id)
                if L >= self.seq_len:
                    idxs = np.sort(rng.choice(L, size=self.seq_len, replace=False))
                else:
                    idxs = np.sort(rng.choice(L, size=self.seq_len, replace=True))

        events_sel = [events[i] for i in idxs]
        times_sel = [times_ts[i] for i in idxs]
        td_sel = np.asarray([time_diff[i] for i in idxs], dtype=np.float32)

        enc = self.tokenizer(
            events_sel,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )

        out: Dict[str, Any] = {
            "input_ids": enc["input_ids"].long(),    
            "times_ts": times_sel,                    
            "time_diff": torch.from_numpy(td_sel),    
            "subject_id": torch.tensor(subject_id, dtype=torch.int64),
            "hadm_id": torch.tensor(hadm_id, dtype=torch.int64),
            "icustay_id": torch.tensor(icustay_id, dtype=torch.int64),
        }

        if self.label_field is not None:
            y = row[self.label_field]
            out["label"] = torch.tensor(float(y), dtype=torch.float32)

        return out
    

class REMedGenHPFCollator:
    def __init__(
        self,
        pad_token_id: int = 0,
        time_mode: str = "timestamp",   
        time_diff_unit: str = "days",   
    ) -> None:
        self.pad_token_id = pad_token_id
        assert time_mode in {"timestamp", "time_diff"}
        self.time_mode = time_mode
        assert time_diff_unit in {"days", "minutes"}
        self.time_diff_unit = time_diff_unit

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [b for b in batch if b.get("input_ids") is not None]
        if len(batch) == 0:
            return {}
        input_ids_list = [b["input_ids"] for b in batch]
        sizes = [x.size(0) for x in input_ids_list]
        B = len(input_ids_list)
        S_max = max(sizes)
        W = input_ids_list[0].size(1)

        collated_input_ids = input_ids_list[0].new_full(
            (B, S_max, W), fill_value=self.pad_token_id
        ).long()

        padding_mask = torch.ones(B, S_max, dtype=torch.bool)
        collated_times = torch.zeros(B, S_max, dtype=torch.float32)

        for i, b in enumerate(batch):
            ids = b["input_ids"]
            S_i = ids.size(0)

            collated_input_ids[i, :S_i, :] = ids
            padding_mask[i, :S_i] = False

            if self.time_mode == "timestamp":
                ts_list = b["times_ts"]
                anchor = None
                for t in ts_list:
                    if t is not None:
                        anchor = t
                        break
                if anchor is None:
                    
                    times_min = torch.zeros(S_i, dtype=torch.float32)
                else:
                    
                    vals = []
                    for t in ts_list:
                        if t is None:
                            vals.append(0.0)
                        else:
                            vals.append((t - anchor).total_seconds() / 60.0)
                    times_min = torch.tensor(vals, dtype=torch.float32)

            else:
                
                td = b["time_diff"].float()
                if self.time_diff_unit == "days":
                    td = td * 1440.0
                
                times_min = torch.cumsum(td, dim=0)
               
                if S_i > 0:
                    times_min = times_min - times_min[0]

            collated_times[i, :S_i] = times_min

        out: Dict[str, Any] = {
            "input_ids": collated_input_ids,
            "padding_mask": padding_mask,
            "times": collated_times,  
        }

        if "label" in batch[0] and batch[0]["label"] is not None:
            out["label"] = torch.stack([b["label"] for b in batch])


        out["subject_id"] = torch.stack(
            [b["subject_id"] for b in batch]
        )

        out["hadm_id"] = torch.stack(
            [b["hadm_id"] for b in batch]
        )

        out["icustay_id"] = torch.stack(
            [b["icustay_id"] for b in batch]
        )

        return out
    

#########################################################
# EHRMamba model
#########################################################

class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        chunks = []

        for item in batch:
            if isinstance(item, dict):
                chunks.append(item)
            else:
                chunks.extend(item)

        keys = [
            k for k in chunks[0].keys()
            if k not in ("text_values",)
        ]

        out = {
            k: torch.stack(
                [torch.as_tensor(c[k]) for c in chunks],
                0
            )
            for k in keys
        }


        # Clean numeric values
        if "numeric_values" in out:

            vals = out["numeric_values"].float()

            finite_mask = torch.isfinite(vals)

            if "numeric_mask" in out:
                mask = out["numeric_mask"].bool() & finite_mask
            else:
                mask = finite_mask

            vals = torch.nan_to_num(vals,nan=0.0,posinf=0.0,neginf=0.0,)

            out["numeric_values"] = vals
            out["numeric_mask"] = mask


        # Clean time features
        if "time_diff" in out:

            t = out["time_diff"].float()

            t = torch.nan_to_num(t, nan=0.0,posinf=0.0,neginf=0.0,)

            out["time_diff"] = t

        # Causal LM labels
        labels = out["input_ids"].clone()
        pad = (self.tokenizer.pad_id if self.tokenizer.pad_id is not None else 0)
        labels[labels == pad] = -100
        if self.tokenizer.cls_id is not None:
            labels[out["input_ids"] == self.tokenizer.cls_id] = -100
        out["labels"] = labels
        return out
    

#########################################################
# HiBEHRT model
#########################################################


class HiBEHRTEvalCollator:
    def __init__(self, seq_gen, chunk_length: int = 256, overlap: int = 32, add_cls_per_chunk: bool = True):
        self.seq_gen = seq_gen
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.add_cls_per_chunk = add_cls_per_chunk

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        chunked_per_patient = []
        labels = []

        for sample in batch:
            labels.append(sample["label"])

            timeline = {k: v for k, v in sample.items() if k != "label"}

            chunks = self.seq_gen.get_overlapped_chunks(
                timeline=timeline,
                chunk_length=self.chunk_length,
                overlap=self.overlap,
                add_cls_per_chunk=self.add_cls_per_chunk,
            )
            chunked_per_patient.append(chunks)

        n_max = max(len(chunks) for chunks in chunked_per_patient)

        keys = list(chunked_per_patient[0][0].keys())

        out = {}
        for k in keys:
            per_patient_tensors = []
            for chunks in chunked_per_patient:
                tensors = [torch.as_tensor(ch[k]) for ch in chunks]
                if len(tensors) < n_max:
                    pad_chunk = torch.zeros_like(tensors[0])
                    tensors.extend([pad_chunk] * (n_max - len(tensors)))
                per_patient_tensors.append(torch.stack(tensors, dim=0))  
            out[k] = torch.stack(per_patient_tensors, dim=0)  

        out["label"] = torch.as_tensor(labels)

        return out