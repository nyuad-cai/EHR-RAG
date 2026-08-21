import gc
import os
import torch
import polars as pl
from src.models.baseline_models import load_hf_model
from  datasets import load_from_disk
from src.models.utils import predict_dataset, compute_metrics_with_ci_llm

import argparse

parser = argparse.ArgumentParser(description='LLM pretraining command line interface')
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--data-idx-path', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
args = parser.parse_args()




model_name=args.model_name
print(model_name)


hf_token = os.getenv("HF_TOKEN")
dataset = load_from_disk(args.dataset_path)

window = 'within_stay_genhpf'
task = 'y_mort_12mo'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=args.data_idx_path, window=window, task_name=task, model_bundle=bundle, split=args.split) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within_stay_genhpf'
task = 'y_icu_readmit_30'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=args.data_idx_path, window=window, task_name=task, model_bundle=bundle, split=args.split) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within48_genhpf'
task = 'y_mort'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=args.data_idx_path, window=window, task_name=task, model_bundle=bundle, split=args.split) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)


window = 'within24_genhpf'
task = 'y_los_7'
print('Running task: ', task)
bundle = load_hf_model(model_name=model_name, hf_token=hf_token)
results = predict_dataset(dataset=dataset, data_idx_path=args.data_idx_path, window=window, task_name=task, model_bundle=bundle, split=args.split) 
metrics = compute_metrics_with_ci_llm(results)
print(metrics)
del bundle
gc.collect()
torch.cuda.empty_cache()
print('Finished task: ', task)

print('All tasks completed!')

