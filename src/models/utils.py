

import os
import yaml
import json
import torch
import wandb
import polars as pl
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist

from typing import Callable
from transformers import BertConfig
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING

def correct_tokenizer_dict(dict_fp:str):
    with open(dict_fp) as f:
        old_dictionary = json.load(f)
    dictionary = {"age_stats": old_dictionary["age_stats"],
                  "is_hierarchical": old_dictionary.get("is_hierarchical", False),
                  "vocab": old_dictionary["regular"]}
    return dictionary

BERT_VARIANTS = {
    "bert": {},
    "medbert": dict(
        hidden_size=192,
        intermediate_size=64,
        num_attention_heads=6,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "cehrbert": dict(
        hidden_size=128,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "behrt": dict(
        hidden_size=288,
        intermediate_size=512,
        num_attention_heads=12,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ),

    "hibehrt": dict(
        hidden_size=150,
        intermediate_size=108,
        num_attention_heads=6,
        num_hidden_layers=4,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.3,
    ),
}



def get_config_and_model_cls(model_type: str, mode: str = "mlm", variant: str = None):

    assert mode in ["mlm", "eval", "causal"]

    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unknown model_type: {model_type}")

    config_cls = CONFIG_MAPPING[model_type]

    if mode == "mlm":
        model_cls = MODEL_FOR_MASKED_LM_MAPPING[config_cls]
    elif mode == "eval":
        model_cls = MODEL_MAPPING[config_cls]
    else:
        model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config_cls]


    variant_kwargs = {}

    if variant is not None and issubclass(config_cls, BertConfig):
        if variant not in BERT_VARIANTS:
            raise ValueError(f"Unknown BERT variant: {variant}")

        variant_kwargs = BERT_VARIANTS[variant]


    def build_config(**kwargs):
        return config_cls(**variant_kwargs, **kwargs)

    return build_config, model_cls


def fix_roberta_longformer_max_pos(cfg):

    model_type = getattr(cfg, "model_type", "").lower()

    if model_type == "roberta":
        if cfg.max_position_embeddings == 512:
            cfg.max_position_embeddings = 513

    elif model_type == "longformer":
        if cfg.max_position_embeddings == 512:
            cfg.max_position_embeddings = 4097

    return cfg



def load_config_with_env(path):
    # read file
    with open(path, "r") as f:
        raw_text = f.read()
    expanded = os.path.expandvars(raw_text)
    
    return yaml.safe_load(expanded)


class Time2Vec(nn.Module):

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 16,
        periodic_activation: Callable = torch.sin,
    ):
        super().__init__()
        assert out_features >= 1, "out_features must be >= 1"

        self.in_features = in_features
        self.out_features = out_features
        self.periodic_activation = periodic_activation

        self.W = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

        self.W0 = nn.Parameter(torch.randn(in_features))
        self.b0 = nn.Parameter(torch.randn(1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:

        v1 = self.periodic_activation(tau @ self.W + self.b)
        v2 = (tau @ self.W0).unsqueeze(-1) + self.b0

        return torch.cat([v2, v1], dim=-1)
    

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()




def get_bootstrap_ci(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    num_iter: int = 1000,
    alpha: float = 0.05,
    ndigits: int = 3,
):
    device = y_score.device

    y_true = y_true.detach().view(-1).to(device).long()
    y_score = y_score.detach().view(-1).to(device)

    auroc_point = BinaryAUROC().to(device)(y_score, y_true)
    auprc_point = BinaryAveragePrecision().to(device)(y_score, y_true)

    n = y_true.numel()
    auroc_samples = torch.empty(num_iter, device=device)
    auprc_samples = torch.empty(num_iter, device=device)

    for i in range(num_iter):
        idx = torch.randint(0, n, (n,), device=device)
        auroc_samples[i] = BinaryAUROC().to(device)(y_score[idx], y_true[idx])
        auprc_samples[i] = BinaryAveragePrecision().to(device)(y_score[idx], y_true[idx])

    # Percentile CI
    q_low = alpha / 2.0         # 2.5%
    q_high = 1.0 - alpha / 2.0  # 97.5%

    auroc_low = torch.quantile(auroc_samples, q_low)
    auroc_high = torch.quantile(auroc_samples, q_high)

    auprc_low = torch.quantile(auprc_samples, q_low)
    auprc_high = torch.quantile(auprc_samples, q_high)

    def _fmt(point, low, high):
        p = float(point.detach().cpu())
        l = float(low.detach().cpu())
        h = float(high.detach().cpu())
        return f"{round(p, ndigits)} ({round(l, ndigits)}, {round(h, ndigits)})"

    auroc_text = _fmt(auroc_point, auroc_low, auroc_high)
    auprc_text = _fmt(auprc_point, auprc_low, auprc_high)

    return auroc_text, auprc_text


def gather_1d_varlen_pl(module, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().view(-1)

    if not getattr(module, "trainer", None) or module.trainer.world_size == 1:
        return x

    device = x.device
    local_len = torch.tensor([x.numel()], device=device, dtype=torch.long)

    all_lens = module.all_gather(local_len).view(-1) 
    max_len = int(all_lens.max().item())

    if x.numel() < max_len:
        pad = torch.zeros(max_len - x.numel(), device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    x_gather = module.all_gather(x_pad)

    chunks = []
    for r in range(x_gather.shape[0]):
        chunks.append(x_gather[r, : int(all_lens[r].item())])
    return torch.cat(chunks, dim=0)


def log_bootstrap_ci_text_percentile(
    module,
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    prefix: str = "test",
    num_iter: int = 1000,
    alpha: float = 0.05,
    ndigits: int = 3,
):
    y_all = gather_1d_varlen_pl(module, y_true)
    s_all = gather_1d_varlen_pl(module, y_score)

    if not getattr(module, "trainer", None) or module.trainer.is_global_zero:
        auroc_ci_text, auprc_ci_text = get_bootstrap_ci(
            y_true=y_all,
            y_score=s_all,
            num_iter=num_iter,
            alpha=alpha,
            ndigits=ndigits,
        )

        # Use wandb.log directly for string-based CI values
        wandb.log({
            f"{prefix}_auroc_ci": auroc_ci_text,
            f"{prefix}_auprc_ci": auprc_ci_text,
        }, commit=False)



#########################################################
# LLM baselines
#########################################################

TASK_PROMPTS = {
    "y_los_7": (
        "You are an expert clinical risk prediction model using electronic health records.\n\n"
        "--- PATIENT DATA ---\n"
        "Electronic Health Records:\n"
        "{ehr_text}\n\n"
        "--- TASK ---\n"
        "will this patient have a hospital length of stay longer than 7 days?\n\n"
        "Answer only using one word: Yes or No.\n\n"
        "Answer:"
    ),
    "y_icu_readmit_30": (
        "You are an expert clinical risk prediction model using electronic health records.\n\n"
        "--- PATIENT DATA ---\n"
        "Electronic Health Records:\n"
        "{ehr_text}\n\n"
        "--- TASK ---\n"
        "Will this patient be readmitted to the ICU within 30 days after ICU discharge?\n\n"
        "Answer only using one word: Yes or No.\n\n"
        "Answer:"
    ),
    "y_mort": (
        "You are an expert clinical risk prediction model using electronic health records.\n\n"
        "--- PATIENT DATA ---\n"
        "Electronic Health Records:\n"
        "{ehr_text}\n\n"
        "--- TASK ---\n"
        "Will this patient die during this hospital admission?\n\n"
        "Answer only using one word: Yes or No.\n\n"
        "Answer:"
    ),
    "y_mort_12mo": (
        "You are an expert clinical risk prediction model using electronic health records.\n\n"
        "--- PATIENT DATA ---\n"
        "Electronic Health Records:\n"
        "{ehr_text}\n\n"
        "--- TASK ---\n"
        "Will this patient die within 1 year after hospital discharge?\n\n"
        "Answer only using one word: Yes or No.\n\n"
        "Answer:"
    ),
}


def build_prediction_prompt(ehr_text: str, task_name: str) -> str:
    if task_name not in TASK_PROMPTS:
        raise ValueError(f"Unknown task_name: {task_name}")
    return TASK_PROMPTS[task_name].format(ehr_text=ehr_text)


def predict_yes_no_probability(ehr_text: str, task_name: str, model_bundle: dict):
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    yes_token_id = model_bundle["yes_token_id"]
    no_token_id = model_bundle["no_token_id"]

    prompt = build_prediction_prompt(ehr_text, task_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[:, -1, :].float()

    yes_logit = next_token_logits[0, yes_token_id]
    no_logit = next_token_logits[0, no_token_id]

    yes_no_logits = torch.stack([no_logit, yes_logit], dim=0)
    probs = torch.softmax(yes_no_logits, dim=0)

    no_prob = float(probs[0].cpu())
    yes_prob = float(probs[1].cpu())

    pred_label = 1 if yes_prob > 0.5 else 0
    pred_text = "Yes" if pred_label == 1 else "No"

    return {
        "prompt": prompt,
        "pred_text": pred_text,
        "pred_label": pred_label,
        "yes_prob": yes_prob,
        "no_prob": no_prob,
    }

def compute_metrics_with_ci_llm(results):

    y_true = torch.tensor([r["ground_truth"] for r in results], dtype=torch.long)
    y_score = torch.tensor([r["yes_prob"] for r in results], dtype=torch.float32)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = y_true.to(device)
    y_score = y_score.to(device)


    auroc = BinaryAUROC().to(device)(y_score, y_true)
    auprc = BinaryAveragePrecision().to(device)(y_score, y_true)


    auroc_ci, auprc_ci = get_bootstrap_ci(y_true, y_score)

    return {
        "AUROC": float(auroc.cpu()),
        "AUPRC": float(auprc.cpu()),
        "AUROC_CI": auroc_ci,
        "AUPRC_CI": auprc_ci,
    }


def predict_dataset(dataset, data_idx_path, window, task_name, model_bundle, split):

    results = []

    idx = pl.read_parquet(data_idx_path)
    test_idx = idx.filter(pl.col("split") == split)

    test_rows = {
        int(row["icustay_id"]): row
        for row in test_idx.to_dicts()
    }

    # build direct lookup once using only the icustay_id column
    icustay_ids = dataset["icustay_id"]

    hf_lookup = {
        int(stay_id): i
        for i, stay_id in enumerate(icustay_ids)
    }

    for stay_id, row in tqdm(test_rows.items()):

        hf_idx = hf_lookup[stay_id]
        sample = dataset[hf_idx]

        query = sample[window]

        prediction = predict_yes_no_probability(
            ehr_text="\n".join(query),
            task_name=task_name,
            model_bundle=model_bundle,
        )

        results.append({
            "subject_id": sample["subject_id"],
            "stay_id": stay_id,
            "ground_truth": row[task_name],
            "yes_prob": prediction["yes_prob"],
            "no_prob": prediction["no_prob"],
        })

    return results