import os
import shutil 

import polars as pl

from tqdm import tqdm
from pathlib import Path
from src.data.utils import *
from src.data.datasets import Tokenizer, SequencesGenerator

"""
This code works on top of already extracted MIMIC-IV in MEDS format.
    to use it you need to first extract the dataset in MEDS format using 
    MIMIC_IV_MEDS[https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS] (V 0.1.2).
    post extract, move the MEDS_output directory to data directory. The preprocessing codes runs a stages,
    the next satge can not start unless the current one finishes
    preprocessing stages performed in this script:

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
"""



# Data directory structure building
data_path = os.path.join('.','data')
raw_meds_data_path = os.path.join(data_path,'MEDS_output','data')
raw_meds_metadata_path = os.path.join(data_path,'MEDS_output','metadata')
splits = [x for x in os.listdir(raw_meds_data_path) if x != ".logs"]

processed_path = os.path.join(data_path,'processed')
processed_data_path = os.path.join(processed_path,'data')
os.makedirs(processed_data_path, exist_ok=True)

for split in splits:
    os.makedirs(os.path.join(processed_data_path, split),exist_ok=True)

#######################################################################
# Processing (1-6)
#######################################################################

for split in tqdm(splits):
    print(f'Working on {split} split\n')

    for file in tqdm(os.listdir(os.path.join(raw_meds_data_path,split))):
        output_file = os.path.join(processed_data_path, split,file)

        if os.path.exists(output_file):
            print(f"Skipping existing: {split}/{file}\n")
            continue
        else:
            print(f'Working on {file} shard\n')
            shard = pl.read_parquet(os.path.join(raw_meds_data_path, split, file))
            # 1.
            shard = shard.filter(pl.col('code').str.starts_with('HCPCS') == False)
            # 2. 
            shard = process_omr_numeric(shard)

            processed = []

            for subject_id in tqdm(shard['subject_id'].unique()):
                
                
                timeline = shard.filter(pl.col('subject_id') == subject_id)
                # 3.
                timeline = segment_care_stage(timeline)
                # 4.
                timeline = assign_visit_id(timeline)
                # 5. 
                timeline = add_time_tokens(timeline)
                # 6.
                timeline = add_token_type(timeline)

                processed.append(timeline)

            shard = pl.concat(processed)
            shard = shard.with_columns(pl.col("numeric_value").alias("numeric_value_original"))
            shard.write_parquet(output_file)
            print(f'Finished {file} shard \n')
    print(f'Finished {split} split \n')
shutil.copytree(raw_meds_metadata_path, os.path.join(processed_path,'metadata'), dirs_exist_ok=True)


#######################################################################
# Outliers elimination (7,8)
#######################################################################


outliers_path = os.path.join(data_path,'meds_outliers')
if check_stage_complete(processed_path, outliers_path,splits):
    print("Skipping outlier removal: stage already completed.")
else:
    os.makedirs(outliers_path,exist_ok=True)
    empty_dir(outliers_path)
    phase1_config["input_dir"] = processed_path
    phase1_config["output_dir"] = outliers_path
    run_meds_transform_from_dict(phase1_config)
    print('Finished outliers/rare-events cleaning\n')

#######################################################################
# Normalization (9)
#######################################################################

normalized_path = os.path.join(data_path, "meds_normalized")
metadata_path = os.path.join(normalized_path, "metadata", "codes.parquet")

if check_stage_complete(outliers_path, normalized_path, splits):
    print("Skipping normalization: stage already completed.")

else:
    os.makedirs(normalized_path, exist_ok=True)
    empty_dir(normalized_path)

    # Compute fresh post-filtering/post-outlier metadata.
    phase2_config["input_dir"] = outliers_path
    phase2_config["output_dir"] = normalized_path

    run_meds_transform_from_dict(phase2_config)

    print("Finished normalization stat computation\n")


    code_metadata = pl.read_parquet(metadata_path)

    mean_expr = (
        pl.col("values/sum")
        / pl.col("values/n_occurrences")
    )

    std_expr = (
        (
            pl.col("values/sum_sqd")
            / pl.col("values/n_occurrences")
        )
        - mean_expr.pow(2)
    ).sqrt()

    normalization_metadata = code_metadata.select(
        "code",
        mean_expr.alias("values/mean"),
        std_expr.alias("values/std"),
    )

    for split in splits:
        input_split = os.path.join(
            outliers_path,
            "data",
            split,
        )

        output_split = os.path.join(
            normalized_path,
            "data",
            split,
        )

        for root, _, files in os.walk(input_split):
            for filename in tqdm(files):
                if not filename.endswith(".parquet"):
                    continue

                input_file = os.path.join(root, filename)

                relative_file = os.path.relpath(
                    input_file,
                    input_split,
                )

                output_file = os.path.join(
                    output_split,
                    relative_file,
                )

                os.makedirs(
                    os.path.dirname(output_file),
                    exist_ok=True,
                )

                print(f"Normalizing {split}/{relative_file}")

                row_idx = "_normalization_row_idx"

                df = pl.scan_parquet(input_file)

                while row_idx in df.collect_schema().names():
                    row_idx = f"_{row_idx}"

                normalized = (
                    df
                    .with_row_index(row_idx)
                    .join(
                        normalization_metadata.lazy(),
                        on="code",
                        how="inner",
                        nulls_equal=True,
                    )
                    .with_columns(
                        (
                            (
                                pl.col("numeric_value")
                                - pl.col("values/mean")
                            )
                            / pl.col("values/std")
                        )
                        .cast(pl.Float32)
                        .alias("numeric_value")
                    )
                    .drop(
                        "values/mean",
                        "values/std",
                    )
                    .sort(row_idx)
                    .drop(row_idx)
                )
                normalized = normalized.sort([
                                                "subject_id",
                                                "event_idx",
                                                "time"
                                            ]
                                        )
                normalized = normalized.with_columns(
                    pl.col("event_idx")
                    .cum_count()
                    .over("subject_id")
                    .sub(1)
                    .alias("event_idx")
                )
                normalized.sink_parquet(output_file)

    print("Finished manual normalization\n")


#######################################################################
# Toeknizer fitting (10)
#######################################################################

resources_path = os.path.join('.','resources')
os.makedirs(resources_path,exist_ok=True)

if os.path.exists(os.path.join(resources_path,'vocab.json')):
    print('vocab already fitted')
else:
    event_types = set()
    split_path = os.path.join(normalized_path, "data", 'train')

    for file in tqdm(os.listdir(split_path)):
        if not file.endswith(".parquet"):
            continue
        df = pl.read_parquet(
            os.path.join(split_path, file),
            columns=["code_type"]
        )

        event_types.update(
            df["code_type"]
            .drop_nulls()
            .unique()
            .to_list()
        )
    event_types = sorted(event_types)


    tokenizer = Tokenizer(codes_parquet_fp=metadata_path,
                        special_tokens=['[PAD]', '[MASK]', '[CLS]', '[UNK]'],
                        event_types=event_types,
                        force_special_ids=True)

    tokenizer.save(os.path.join(resources_path,'vocab.json'))


#######################################################################
# Pretraining index buidling (11)
#######################################################################

pretrain_index_path = os.path.join(resources_path, "pretrain_index.parquet")

if os.path.exists(pretrain_index_path):
    print("Pretrain index already exists.")

else:

    train_path = os.path.join(
        normalized_path,
        "data",
        "train"
    )

    index_rows = []

    for shard in tqdm(os.listdir(train_path)):

        if not shard.endswith(".parquet"):
            continue

        shard_path = os.path.join(train_path,shard)

        df = pl.read_parquet(shard_path, columns=["subject_id"])

        counts = (
            df
.group_by("subject_id")
            .len()
            .rename({"len": "n_events"})
            .with_columns(
                pl.lit(shard).alias("shard"),
                pl.lit("train").alias("split"),
            )
        )

        index_rows.append(counts)


    pretrain_index = pl.concat(index_rows)


    # -------------------------
    # Create MLM validation split (5%)
    # -------------------------
    mlm_val_subjects = (
        pretrain_index
        .select("subject_id")
        .sample(
            fraction=0.05,
            seed=42
        )
        .get_column("subject_id")
        .to_list()
    )


    pretrain_index = pretrain_index.with_columns(
        pl.when(
            pl.col("subject_id").is_in(mlm_val_subjects)
        )
        .then(pl.lit("tuning"))
        .otherwise(pl.col("split"))
        .alias("split")
    )


    pretrain_index.write_parquet(pretrain_index_path)

    print("Pretrain index created.")

#######################################################################
# Arrow dataset building (12)
#######################################################################


arrow_path = os.path.join(data_path, "meds_normalized_arrow")
arrow_done = os.path.join(arrow_path, ".done")


if os.path.exists(arrow_done):

    print("Skipping Arrow dataset creation: already completed.")

else:

    seq_gen = SequencesGenerator(
        tokenizer_path=os.path.join(resources_path, "vocab.json"),
        chunk_length=1024, # this is just set for the API correctness, full sequnece will be encoded
        overlap=128, # this is just set for the API correctness, full sequnece will be encoded
        return_numeric=True,
        return_text=True,
        return_time=True,
        return_ids=True,
    )


    build_arrow_dataset(
        normalized_data_dir=normalized_path,
        writer_batch_size=100,
        splits=splits,
        output_dir=arrow_path,
        seq_gen=seq_gen,
    )
    # checkpoint only after successful completion
    Path(arrow_done).touch()

    print("Finished Arrow dataset creation.")



#######################################################################
# Downstream cohort filtering (13)
# Ground truth labels extraction (14)
# Query/History boundaries identification (15)
#######################################################################

icu_downstream_index_path = os.path.join(resources_path, "downstream_index.parquet")
inpat_downstream_index_path = os.path.join(resources_path, "inpatient_index.parquet")
raw_downstream_index_path = os.path.join(resources_path, "raw_index.parquet")

if all(os.path.exists(p) for p in [icu_downstream_index_path, inpat_downstream_index_path, raw_downstream_index_path]):
    print('downstream indices are already done')
else:
    # Downstream cohort filtering (13)
    downstream_idx = build_stay_index(normalized_path=normalized_path,splits=splits)
    # Exclude patients with no hadm_id
    downstream_idx = downstream_idx.filter(pl.col('hadm_id').is_not_null())
    # Exclued admissions with age at admission < 18 years old
    downstream_idx = downstream_idx.filter(pl.col('age_at_admission')>= 18)
    # Split cohorts into icu and inpatients admisions
    icu_downstream_idx = downstream_idx.filter(pl.col('icustay_id').is_not_null())
    inpatient_downstream_idx= downstream_idx.filter(pl.col('icustay_id').is_null())
    # Keep hospital admission with single ICU stay only
    valid_hadm = (
        icu_downstream_idx
        .group_by(["subject_id", "hadm_id"])
        .agg(
            pl.col("icustay_id").n_unique().alias("n_icu")
        )
        .filter(
            pl.col("n_icu") == 1
        )
        .select(
            ["subject_id", "hadm_id"]
        )
    )

    icu_downstream_idx = icu_downstream_idx.join(valid_hadm,on=["subject_id", "hadm_id"],how="inner")
    # patients whose admission and discharge ICU units are identical
    icu_downstream_idx = icu_downstream_idx.filter(
        pl.col("icu_adm_loc").is_not_null()
        &
        pl.col("icu_disch_loc").is_not_null()
        &
        (pl.col("icu_adm_loc") == pl.col("icu_disch_loc"))
    )
    # keep stays with mimimu los of 24 hrs
    icu_downstream_idx = icu_downstream_idx.filter(
        pl.col("icu_los").is_not_null()
        &
        (pl.col("icu_los") >= 1)
    )
    # Ground truth labels extraction (14)

    icu_downstream_idx = get_mortality_labels(icu_downstream_idx)

    icu_downstream_idx = get_los_labels(icu_downstream_idx, durations=[7, 15, 30])

    icu_downstream_idx = get_post_discharge_mortality_labels(icu_downstream_idx, months=[1, 3, 6, 9, 12])

    icu_downstream_idx = get_icu_readmission_labels(icu_downstream_idx, windows=[7, 15, 30])

    # Query/History boundaries identification (15)
    window_cols = [
        "w24_min",
        "w24_max",
        "w48_min",
        "w48_max",
        "wStay_min",
        "wStay_max",
    ]

    icu_downstream_idx = icu_downstream_idx.with_columns(
        [
            pl.col("icu_adm_idx").alias("w24_min"),
            pl.col("icu_adm_idx").alias("w48_min"),
            pl.col("icu_adm_idx").alias("wStay_min"),
            pl.col("hosp_disch_idx").alias("wStay_max"),
        ]
    )

    other_cols = [
        c for c in icu_downstream_idx.columns
        if c not in window_cols
    ]

    icu_downstream_idx = icu_downstream_idx.select(
        other_cols + window_cols
    )


    # add leakage safe query/history boundaries 
    icu_downstream_idx = add_query_boundaries(
        icu_downstream_idx
    )

    contexts = [512, 1024, 2048]
    windows = ["w24", "w48", "wStay"]

    conditions = []

    for w in windows:
        for c in contexts:
            conditions.append(
                pl.col(f"{w}_start_{c}") <= pl.col(f"{w}_end_{c}")
            )

    icu_downstream_idx = icu_downstream_idx.filter(
        pl.all_horizontal(conditions)
    )

    for window in windows:
        for context in contexts:
            assert (
                icu_downstream_idx
                .filter(
                    pl.col(f"{window}_start_{context}") >
                    pl.col(f"{window}_end_{context}")
                )
                .height
                == 0
            ), f"Invalid boundary found for {window}_{context}"
    print('index created successfully')

    icu_downstream_idx.write_parquet(icu_downstream_index_path)
    inpatient_downstream_idx.write_parquet(inpat_downstream_index_path)
    downstream_idx.write_parquet(raw_downstream_index_path)
    

