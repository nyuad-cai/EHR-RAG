
import os
import argparse
from src.data.datasets import limits
from src.vectordb.vectordbs import build_indices


parser = argparse.ArgumentParser()

parser.add_argument("--data-idx-path", type=str, required=True)
parser.add_argument("--hf-dataset-path", type=str, required=True)
parser.add_argument("--tokenizer-path", type=str, required=True)
parser.add_argument("--ckpt-path", type=str, required=True)
parser.add_argument("--storage-path", type=str, required=True)

parser.add_argument("--embedder-model", type=str, default="roformer")

parser.add_argument("--use-type", action="store_true")
parser.add_argument("--use-visit", action="store_true")
parser.add_argument("--use-stage", action="store_true")
parser.add_argument("--use-time", action="store_true")
parser.add_argument("--use-numeric", action="store_true")

parser.add_argument("--seq-length-q", type=int, default=1024)
parser.add_argument("--overlap-q", type=int, default=0)

args = parser.parse_args()

windows = [
    ('within24_query', 'within24_hist_full', 'w24'),
    ('within48_query', 'within48_hist_full', 'w48'),
    ('within_stay_query', 'within_stay_hist_full', 'wstay'),
]

history_settings = [
    {'chunking_strategy': 'overlap', 'seq_length_h': 256,  'overlap_h': 32,  'window_hours': 6.0},
    {'chunking_strategy': 'overlap', 'seq_length_h': 512,  'overlap_h': 64,  'window_hours': 6.0},
    {'chunking_strategy': 'overlap', 'seq_length_h': 1024, 'overlap_h': 128, 'window_hours': 6.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 6.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 12.0},
    {'chunking_strategy': 'time',    'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 24.0},
    {'chunking_strategy': 'visit',   'seq_length_h': 256,  'overlap_h': 0,   'window_hours': 6.0},
    {'chunking_strategy': 'care_stage', 'seq_length_h': 256, 'overlap_h': 0, 'window_hours': 6.0},
]

for cfg in history_settings:
    for main_window_q, main_window_h, window in windows:
        if cfg['chunking_strategy'] == 'time':
            span_dir = str(cfg['window_hours'])
        else:
            span_dir = str(cfg['seq_length_h'])

        save_path = os.path.join(args.storage_path ,str(args.seq_length_q), cfg['chunking_strategy'], str(span_dir), window)

        os.makedirs(save_path, exist_ok=True)

        print(f"Running: strategy={cfg['chunking_strategy']}, span={span_dir}, window={window}")

        build_indices(
            data_idx_path=args.data_idx_path,
            hf_dataset_path=args.hf_dataset_path,
            tokenizer_path=args.tokenizer_path,
            save_path=save_path,
            main_window_q=main_window_q,
            main_window_h=main_window_h,
            limits_dict=limits,
            ckpt_path=args.ckpt_path,
            embedder_model=args.embedder_model,
            chunking_strategy=cfg['chunking_strategy'],
            seq_length_q=args.seq_length_q,
            overlap_q=args.overlap_q,
            seq_length_h=cfg['seq_length_h'],
            overlap_h=cfg['overlap_h'],
            window_hours=cfg['window_hours'],
            use_type=args.use_type,
            use_visit=args.use_visit,
            use_stage=args.use_stage,
            use_time=args.use_time,
            use_numeric=args.use_numeric,
        )