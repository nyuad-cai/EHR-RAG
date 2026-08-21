import os
import yaml
import torch
import shutil
import tempfile
import subprocess

import polars as pl
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, Features, Sequence, Value
from typing import Any, List, Dict, Union, Optional, Mapping 





OMR_EVENTS = {
    "Blood Pressure Standing (1 min)",
    "Blood Pressure Lying",
    "Blood Pressure Sitting",
    "BMI (kg/m2)",
    "Weight",
    "Height",
    "Height (Inches)",
    "eGFR",
    "Blood Pressure Standing",
    "Weight (Lbs)",
    "Blood Pressure Standing (3 mins)",
    "BMI",
    "Blood Pressure",
}


def segment_care_stage(df: pl.DataFrame) -> pl.DataFrame:

    # chronological order
    df = (
        df
        .sort("time")
        .with_row_index("event_idx")
    )

    care_stages = []

    current_stage = "OUTPATIENT"

    discharge_date = None


    for row in df.iter_rows(named=True):

        code = row["code"]
        hadm_id = row["hadm_id"]
        event_time = row["time"]


        # -------------------------
        # Static events
        # -------------------------
        if row["event_idx"] < 2:

            care_stages.append(None)
            continue


        # -------------------------
        # Close inpatient after
        # discharge day ends
        # -------------------------
        if discharge_date is not None and event_time is not None:

            if event_time.date() > discharge_date:

                if current_stage in ["ED", "INPATIENT", "ICU"]:
                    current_stage = "OUTPATIENT"

                discharge_date = None


        # -------------------------
        # ED start
        # -------------------------
        if code.startswith("ED_REGISTRATION"):

            current_stage = "ED"
            discharge_date = None


        # -------------------------
        # Hospital admission
        # -------------------------
        elif code.startswith("HOSPITAL_ADMISSION"):

            current_stage = "INPATIENT"
            discharge_date = None


        # -------------------------
        # ICU admission
        # -------------------------
        elif code.startswith("ICU_ADMISSION"):

            current_stage = "ICU"


        # -------------------------
        # ICU discharge
        # -------------------------
        elif code.startswith("ICU_DISCHARGE"):

            current_stage = "INPATIENT"


        # -------------------------
        # Hospital discharge
        # Keep inpatient for
        # discharge-day events
        # -------------------------
        elif code.startswith("HOSPITAL_DISCHARGE"):

            current_stage = "INPATIENT"

            if event_time is not None:
                discharge_date = event_time.date()


        # -------------------------
        # OMR events
        # Only switch to outpatient
        # if not inside active encounter
        # -------------------------
        elif code in OMR_EVENTS:

            if current_stage not in ["ED", "INPATIENT", "ICU"]:

                current_stage = "OUTPATIENT"


        # -------------------------
        # Other events
        # Do not modify stage
        # -------------------------
        else:

            pass


        care_stages.append(current_stage)


    return (
        df
        .with_columns(
            pl.Series("care_stage", care_stages)
        )
    )





def assign_visit_id(df: pl.DataFrame) -> pl.DataFrame:

    df = df.sort("time")

    visit_ids = []

    current_visit = 0

    in_acute = False
    last_outpatient_date = None


    for row in df.iter_rows(named=True):

        code = row["code"]
        stage = row["care_stage"]
        event_time = row["time"]


        # -------------------------
        # Static
        # -------------------------
        if stage == None:
            visit_ids.append(None)
            continue


        # -------------------------
        # New ED visit
        # -------------------------
        if code.startswith("ED_REGISTRATION"):

            current_visit += 1
            in_acute = True

            visit_ids.append(current_visit)
            continue


        # -------------------------
        # New inpatient visit
        # only if not already in ED
        # -------------------------
        if code.startswith("HOSPITAL_ADMISSION"):

            if not in_acute:
                current_visit += 1

            in_acute = True

            visit_ids.append(current_visit)
            continue


        # -------------------------
        # ICU stays belong to same visit
        # -------------------------
        if stage == "ICU":

            visit_ids.append(current_visit)
            continue


        # -------------------------
        # Remaining inpatient/ED
        # -------------------------
        if stage in ["ED", "INPATIENT"]:

            visit_ids.append(current_visit)
            continue


        # -------------------------
        # Outpatient visits
        # grouped by calendar date
        # -------------------------
        if stage == "OUTPATIENT":

            event_date = (
                event_time.date()
                if event_time is not None
                else None
            )

            if in_acute:
                current_visit += 1
                in_acute = False
                last_outpatient_date = event_date

            elif event_date != last_outpatient_date:

                current_visit += 1
                last_outpatient_date = event_date


            visit_ids.append(current_visit)
            continue


        visit_ids.append(current_visit)


    return (
        df
        .with_columns(
            pl.Series(
                "visit_id",
                visit_ids,
                dtype=pl.Int32
            )
        )
    )


def within(duration: float):

    if duration <= 0:
        return '0-D'

    # exact days
    elif duration > 0 and duration <= 1:
        return '1-D'
    elif duration > 1 and duration <= 2:
        return '2-D'
    elif duration > 2 and duration <= 3:
        return '3-D'
    elif duration > 3 and duration <= 4:
        return '4-D'
    elif duration > 4 and duration <= 5:
        return '5-D'
    elif duration > 5 and duration <= 6:
        return '6-D'
    elif duration > 6 and duration <= 7:
        return '7-D'

    # weeks
    elif duration > 7 and duration <= 14:
        return '2-W'
    elif duration > 14 and duration <= 21:
        return '3-W'
    elif duration > 21 and duration <= 28:
        return '4-W'

    # months
    elif duration > 28 and duration <= 60:
        return '2-M'
    elif duration > 60 and duration <= 90:
        return '3-M'
    elif duration > 90 and duration <= 120:
        return '4-M'
    elif duration > 120 and duration <= 150:
        return '5-M'
    elif duration > 150 and duration <= 180:
        return '6-M'
    elif duration > 180 and duration <= 210:
        return '7-M'
    elif duration > 210 and duration <= 240:
        return '8-M'
    elif duration > 240 and duration <= 270:
        return '9-M'
    elif duration > 270 and duration <= 300:
        return '10-M'
    elif duration > 300 and duration <= 330:
        return '11-M'
    elif duration > 330 and duration <= 360:
        return '12-M'

    else:
        return '1-Y+'





def add_time_tokens(df: pl.DataFrame) -> pl.DataFrame:

    df = (
        df
        .sort("time")
        .with_columns(
            (
                pl.col("time")
                .diff()
                .dt.total_seconds()
                / (24 * 3600)
            )
            .alias("time_diff")
        )
    )

    rows = []

    previous_visit = None
    previous_visit_end = None
    first_clinical_event = True

    for row in df.iter_rows(named=True):

        visit_id = row["visit_id"]
        care_stage = row["care_stage"]
        current_time = row["time"]

        # keep static tokens
        if row["code"].startswith(("GENDER", "MEDS_BIRTH")):

            rows.append(row)

            # use MEDS_BIRTH as temporal anchor
            if row["time"] is not None:
                previous_visit_end = row["time"]

            continue


        # first clinical event after static section
        if first_clinical_event:

            if (
                current_time is not None 
                and previous_visit_end is not None
            ):

                duration = (
                    current_time - previous_visit_end
                ).total_seconds() / (24 * 3600)

                bucket = within(duration)

                token = row.copy()

                token["code"] = f"TIME-GAP//{bucket}"

                token["care_stage"] = None
                token["visit_id"] = None
                token["hadm_id"] = None

                if "code_type" in token:
                    token["code_type"] = "TIME-GAP"

                if "text_value" in token:
                    token["text_value"] = bucket

                if "numeric_value" in token:
                    token["numeric_value"] = duration

                if "time_diff" in token:
                    token["time_diff"] = None

                rows.append(token)

            first_clinical_event = False


        # normal visit transition
        if previous_visit is not None and visit_id != previous_visit:

            if (
                current_time is not None 
                and previous_visit_end is not None
            ):

                duration = (
                    current_time - previous_visit_end
                ).total_seconds() / (24 * 3600)

                bucket = within(duration)

                token = row.copy()

                token["code"] = f"TIME-GAP//{bucket}"
                token["care_stage"] = None
                token["visit_id"] = None
                token["hadm_id"] = None

                if "code_type" in token:
                    token["code_type"] = "TIME-GAP"

                if "text_value" in token:
                    token["text_value"] = bucket

                if "numeric_value" in token:
                    token["numeric_value"] = duration

                if "time_diff" in token:
                    token["time_diff"] = None

                rows.append(token)


        rows.append(row)

        previous_visit = visit_id
        previous_visit_end = current_time


    return pl.DataFrame(rows, schema=df.schema, strict=False)




def add_token_type(df: pl.DataFrame) -> pl.DataFrame:

    def get_token_type(code):

        if code is None:
            return None

        parts = code.split("//")

        # LAB
        if parts[0] == "LAB":
            if len(parts) > 1:
                if parts[1] == "SPECIMEN_COLLECTED":
                    return "LAB_SPECIMEN"

                elif parts[1] == "RESULT":
                    return "LAB_RESULT"

                else:
                    return "ICU_CHART_EVENT"


        # PROCEDURE
        if parts[0] == "PROCEDURE":

            if len(parts) > 1:
                if parts[1] == "ICD":
                    return "PROCEDURE_ICD"

                elif parts[1] == "START":
                    return "ICU_PROCEDURE_START"

                elif parts[1] == "END":
                    return "ICU_PROCEDURE_END"


        # DIAGNOSIS
        if parts[0] == "DIAGNOSIS":

            if len(parts) > 1 and parts[1] == "ICD":
                return "DIAGNOSIS_ICD"


        # INFUSION
        if parts[0] == "INFUSION_START":
            return "ICU_INFUSION_START"

        if parts[0] == "INFUSION_END":
            return "ICU_INFUSION_END"


        # FLUID OUTPUT
        if parts[0] == "SUBJECT_FLUID_OUTPUT":
            return "ICU_SUBJECT_FLUID_OUTPUT"


        # MEDICATION
        if parts[0] == "MEDICATION":

            if len(parts) > 1:

                if parts[1] == "START":
                    return "MEDICATION_START"

                elif parts[1] == "STOP":
                    return "MEDICATION_STOP"


        return parts[0]

    
    df = df.with_columns(
        pl.col("code")
        .map_elements(
            get_token_type,
            return_dtype=pl.String
        )
        .alias("code_type")
    )
    
    df = (
    df
    .drop("event_idx", strict=False)
    .with_row_index("event_idx")
    )
    return df


def process_omr_numeric(df: pl.DataFrame) -> pl.DataFrame:

    bp_parts = (
        pl.col("text_value")
        .str.split("/")
    )

    bp_sys = (
        bp_parts
        .list.get(0, null_on_oob=True)
        .cast(pl.Float64, strict=False)
    )

    bp_dia = (
        bp_parts
        .list.get(1, null_on_oob=True)
        .cast(pl.Float64, strict=False)
    )

    return df.with_columns(
        pl.when(
            pl.col("code").is_in(OMR_EVENTS)
            & pl.col("code").str.contains("Blood Pressure")
        )
        .then(
            (bp_sys + 2 * bp_dia) / 3
        )

        .when(
            pl.col("code").is_in(OMR_EVENTS)
        )
        .then(
            pl.col("text_value")
            .cast(pl.Float64, strict=False)
        )

        .otherwise(
            pl.col("numeric_value")
        )
        .alias("numeric_value")
    )


def run_meds_transform_from_dict(config: dict):

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False
    ) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        config_fp = f.name

    try:
        subprocess.run(
            [
                "MEDS_transform-pipeline",
                config_fp,
            ],
            check=True,
        )

    finally:
        os.remove(config_fp)

# MEDS-tranform configs phase1: basic metadata aggregation and outlier occlusion
phase1_config = {
    "input_dir": None,
    "output_dir": None,
    "stages": [
        {
            "aggregate_code_metadata": {
                "aggregations": [
                    "values/n_occurrences",
                    "values/sum",
                    "values/sum_sqd",
                    "values/min",
                    "values/max",
                    "code/n_occurrences",
                    "code/n_subjects",
                ]
            }
        },
        {
            "filter_measurements": {
                "_match_revise": [
                    {
                        "_matcher": {
                            "time": {
                                "present": False
                            }
                        }
                    },
                    {
                        "_matcher": {
                            "code": {
                                "regex": "MEDS_DEATH.*|MEDS_BIRTH.*|.*ADMISSION.*|.*DISCHARGE.*|.*TRANSFER_TO.*|DRG.*|LAB.*|INFUSION_START.*|INFUSION_END.*|ICU_.*"
                            }
                        }
                    },
                    {
                        "_matcher": {
                            "time": {
                                "present": True
                            }
                        },
                        "min_occurrences_per_code": 4
                    }
                ]
            }
        },
        {
            "occlude_outliers": {
                "stddev_cutoff": 3,
            }
        },
    ],
}
# MEDS-transform configs phase2: advanced metadata, vocab indexing, normalization
phase2_config = {
    "input_dir": None,
    "output_dir": None,
    "stages": [
        {
            "aggregate_code_metadata": {
                "aggregations": [
                    "values/n_occurrences",
                    "values/sum",
                    "values/sum_sqd",
                    "code/n_occurrences",
                    "code/n_subjects",
                    "values/min",
                    "values/max",
                ]
            }
        },
        # "fit_vocabulary_indices",
        # {
        #     "fit_normalization": {
        #         "_base_stage": "aggregate_code_metadata",
        #         "aggregations": [
        #             "values/n_occurrences",
        #             "values/sum",
        #             "values/sum_sqd",
        #         ],
        #     }
        # },
        # "normalization",
    ],
}

def check_stage_complete(input_dir, output_dir, splits):

    if not os.path.exists(output_dir):
        return False

    for split in splits:

        input_split = os.path.join(input_dir, "data", split)
        output_split = os.path.join(output_dir, "data", split)

        if not os.path.exists(output_split):
            return False

        input_files = [
            f for f in os.listdir(input_split)
            if f.endswith(".parquet")
        ]

        output_files = [
            f for f in os.listdir(output_split)
            if f.endswith(".parquet")
        ]

        if len(input_files) != len(output_files):
            return False

    return True




def build_arrow_dataset(
    normalized_data_dir,
    output_dir,
    seq_gen,
    splits=("train", "tuning", "held_out"),
    writer_batch_size=1000,
):

    features = Features({
        "subject_id": Value("int32"),
        "split": Value("string"),

        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "visit_ids": Sequence(Value("int16")),
        "stage_ids": Sequence(Value("int8")),
        "type_ids": Sequence(Value("int16")),

        "numeric_values": Sequence(Value("float32")),
        "numeric_mask": Sequence(Value("int8")),

        "text_values": Sequence(Value("string")),
        "text_mask": Sequence(Value("int8")),

        "time_diff": Sequence(Value("float32")),
        "time_stamp": Sequence(Value("timestamp[s]")),

        "visit_id": Sequence(Value("int16")),
        "event_idx": Sequence(Value("int32")),
    })


    def gen():

        current_shard = None
        shard_df = None

        for split in splits:

            split_dir = os.path.join(
                normalized_data_dir,
                "data",
                split,
            )

            for shard in os.listdir(split_dir):

                if not shard.endswith(".parquet"):
                    continue

                print(f"Processing {split}/{shard}")

                if shard != current_shard:

                    shard_df = pl.read_parquet(
                        os.path.join(split_dir, shard)
                    )

                    current_shard = shard


                for subject_id in shard_df["subject_id"].unique():

                    timeline = (
                        shard_df
                        .filter(
                            pl.col("subject_id") == subject_id
                        )
                        .sort("event_idx")
                    )


                    encoded = seq_gen.encode_sequence(
                        timeline
                    )


                    yield {
                            "subject_id": int(subject_id),
                            "split": split,

                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded["attention_mask"],
                            "visit_ids": encoded["visit_ids"],
                            "stage_ids": encoded["stage_ids"],
                            "type_ids": encoded["type_ids"],

                            "numeric_values": encoded["numeric_values"],
                            "numeric_mask": encoded["numeric_mask"],

                            "text_values": encoded["text_values"],
                            "text_mask": encoded["text_mask"],

                            "time_diff": encoded["time_diff"],
                            "time_stamp": encoded["time_stamp"],

                            "visit_id": encoded["visit_id"],
                            "event_idx": encoded["event_idx"],
                        }


    ds_arrow = Dataset.from_generator(
        gen,
        features=features,
        writer_batch_size=writer_batch_size,
    )

    ds_arrow.save_to_disk(output_dir)



def empty_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()




def pack_clmbr_chunks(
    chunks: List[Dict[str, List[Any]]],
    patient_id: int = 0,
) -> Dict[str, Any]:

    if len(chunks) == 0:
        raise ValueError("Cannot pack empty chunk list.")

    lengths = [len(c["tokens"]) for c in chunks]

    if any(length == 0 for length in lengths):
        raise ValueError("Empty CLMBR chunk found.")

    total_len = sum(lengths)

    def flatten(key):
        return sum([list(c[key]) for c in chunks], [])

    return {
        "num_patients": len(chunks),
        "num_indices": 0,
        "patient_ids": torch.tensor(
            [patient_id] * total_len,
            dtype=torch.long,
        ),
        "offsets": torch.zeros(len(chunks), dtype=torch.int32),
        "transformer": {
            "tokens": torch.tensor(flatten("tokens"), dtype=torch.long),
            "valid_tokens": torch.tensor(flatten("valid_tokens"), dtype=torch.bool),
            "ages": torch.tensor(flatten("ages"), dtype=torch.float32),
            "normalized_ages": torch.tensor(flatten("normalized_ages"), dtype=torch.float16),
            "timestamps": torch.tensor(flatten("timestamps"), dtype=torch.long),
            "patient_lengths": torch.tensor(lengths, dtype=torch.int32),
            "label_indices": torch.tensor([], dtype=torch.int32),
        },
    }

def get_mortality_labels(index_df):
    return index_df.with_columns(
        pl.col("hospital_mortality")
        .cast(pl.Int8)
        .alias("y_mort")
    )


def get_los_labels(index_df, durations):
    expressions = [
        (pl.col("icu_los") >= duration)
        .cast(pl.Int8)
        .alias(f"y_los_{duration}")
        for duration in durations
    ]

    return index_df.with_columns(expressions)



def get_post_discharge_mortality_labels(index_df, months):
    expressions = []

    for month in months:
        horizon_days = (365 / 12) * month

        expr = (
            (
                (pl.col("hospital_mortality") == 0)
                &
                pl.col("meds_death_time").is_not_null()
                &
                (
                    (pl.col("meds_death_time") - pl.col("hosp_disch_time"))
                    > pl.duration(days=0)
                )
                &
                (
                    (pl.col("meds_death_time") - pl.col("hosp_disch_time"))
                    <= pl.duration(days=horizon_days)
                )
            )
            .cast(pl.Int8)
            .alias(f"y_mort_{month}mo")
        )

        expressions.append(expr)

    return index_df.with_columns(expressions)



def get_icu_readmission_labels(index_df, windows):
    index_df = (
        index_df
        .sort(["subject_id", "icu_adm_time"])
        .with_columns(
            pl.col("icu_adm_time")
            .shift(-1)
            .over("subject_id")
            .alias("next_icu_admit")
        )
        .with_columns(
            (
                pl.col("next_icu_admit")
                -
                pl.col("hosp_disch_time")
            )
            .alias("gap_to_next_icu")
        )
    )

    expressions = []

    for window in windows:
        expressions.append(
            (
                pl.col("next_icu_admit").is_not_null()
                &
                pl.col("hosp_disch_time").is_not_null()
                &
                (pl.col("gap_to_next_icu") > pl.duration(days=0))
                &
                (pl.col("gap_to_next_icu") <= pl.duration(days=window))
            )
            .cast(pl.Int8)
            .alias(f"y_icu_readmit_{window}")
        )

    return (
        index_df
        .with_columns(expressions)
        .drop(
            [
                "next_icu_admit",
                "gap_to_next_icu",
            ]
        )
    )





def build_stay_index(normalized_path, splits):

    rows = []

    for split in splits:

        split_dir = os.path.join(
            normalized_path,
            "data",
            split,
        )

        for shard in tqdm(
            os.listdir(split_dir),
            desc=f"Processing {split}"
        ):

            if not shard.endswith(".parquet"):
                continue

            fp = os.path.join(
                split_dir,
                shard
            )

            df = pl.read_parquet(
                fp,
                columns=[
                    "subject_id",
                    "hadm_id",
                    "icustay_id",
                    "time",
                    "event_idx",
                    "code_type",
                    "code",
                ],
            )

            # --------------------------------------------------
            # Raw identifier combinations
            # --------------------------------------------------
            stay_index = (
                df
                .select(
                    [
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                    ]
                )
                .unique()
            )


            # --------------------------------------------------
            # Remove patient-level rows when admission exists
            # --------------------------------------------------
            patients_with_hadm = (
                stay_index
                .filter(
                    pl.col("hadm_id").is_not_null()
                )
                .select("subject_id")
                .unique()
            )

            stay_index = (
                stay_index
                .join(
                    patients_with_hadm.with_columns(
                        pl.lit(True).alias("_has_hadm")
                    ),
                    on="subject_id",
                    how="left",
                )
                .filter(
                    ~(
                        pl.col("hadm_id").is_null()
                        &
                        pl.col("_has_hadm").fill_null(False)
                    )
                )
                .drop("_has_hadm")
            )


            # --------------------------------------------------
            # Remove admission-level rows when ICU exists
            # --------------------------------------------------
            admissions_with_icu = (
                stay_index
                .filter(
                    pl.col("icustay_id").is_not_null()
                )
                .select(
                    [
                        "subject_id",
                        "hadm_id",
                    ]
                )
                .unique()
            )

            stay_index = (
                stay_index
                .join(
                    admissions_with_icu.with_columns(
                        pl.lit(True).alias("_has_icu")
                    ),
                    on=[
                        "subject_id",
                        "hadm_id",
                    ],
                    how="left",
                )
                .filter(
                    ~(
                        pl.col("icustay_id").is_null()
                        &
                        pl.col("_has_icu").fill_null(False)
                    )
                )
                .drop("_has_icu")
            )


            # --------------------------------------------------
            # Birth time
            # --------------------------------------------------
            birth_times = (
                df
                .group_by("subject_id")
                .agg(
                    pl.col("time")
                    .filter(
                        pl.col("code_type") == "MEDS_BIRTH"
                    )
                    .first()
                    .alias("birth_time")
                )
            )


            # --------------------------------------------------
            # Hospital boundaries
            # --------------------------------------------------

            hosp_bounds = (
                df
                .filter(
                    pl.col("hadm_id").is_not_null()
                )
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                    ]
                )
                .agg(
                    [
                        pl.col("time")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_ADMISSION"
                        )
                        .first()
                        .alias("hosp_adm_time"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_ADMISSION"
                        )
                        .first()
                        .alias("hosp_adm_idx"),

                        pl.col("code")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_ADMISSION"
                        )
                        .str.split("//")
                        .list.last()
                        .first()
                        .alias("hosp_adm_loc"),


                        pl.col("time")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_DISCHARGE"
                        )
                        .first()
                        .alias("hosp_disch_time"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_DISCHARGE"
                        )
                        .first()
                        .alias("hosp_disch_idx"),

                        pl.col("code")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_DISCHARGE"
                        )
                        .str.split("//")
                        .list.last()
                        .first()
                        .alias("hosp_disch_loc"),
                    ]
                )
            )


            # --------------------------------------------------
            # ICU boundaries
            # --------------------------------------------------
            icu_bounds = (
                df
                .filter(
                    pl.col("icustay_id").is_not_null()
                )
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                    ]
                )
                .agg(
                    [
                        pl.col("time")
                        .filter(
                            pl.col("code_type") == "ICU_ADMISSION"
                        )
                        .first()
                        .alias("icu_adm_time"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "ICU_ADMISSION"
                        )
                        .first()
                        .alias("icu_adm_idx"),

                        pl.col("code")
                        .filter(
                            pl.col("code_type") == "ICU_ADMISSION"
                        )
                        .str.split("//")
                        .list.last()
                        .first()
                        .alias("icu_adm_loc"),


                        pl.col("time")
                        .filter(
                            pl.col("code_type") == "ICU_DISCHARGE"
                        )
                        .first()
                        .alias("icu_disch_time"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "ICU_DISCHARGE"
                        )
                        .first()
                        .alias("icu_disch_idx"),

                        pl.col("code")
                        .filter(
                            pl.col("code_type") == "ICU_DISCHARGE"
                        )
                        .str.split("//")
                        .list.last()
                        .first()
                        .alias("icu_disch_loc"),
                    ]
                )
            )


            # --------------------------------------------------
            # ICU observation window boundaries with discharge fallback
            # --------------------------------------------------

            icu_window_raw = (
                df
                .filter(
                    pl.col("icustay_id").is_not_null()
                )
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                    ]
                )
                .agg(
                    [
                        pl.col("time")
                        .filter(
                            pl.col("code_type") == "ICU_ADMISSION"
                        )
                        .first()
                        .alias("icu_start_time"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "ICU_ADMISSION"
                        )
                        .first()
                        .alias("icu_adm_idx"),

                        (
                            pl.col("event_idx")
                            .filter(
                                pl.col("time")
                                <=
                                (
                                    pl.col("time")
                                    .filter(
                                        pl.col("code_type") == "ICU_ADMISSION"
                                    )
                                    .first()
                                    +
                                    pl.duration(hours=24)
                                )
                            )
                            .max()
                        )
                        .alias("_w24_time_max"),

                        (
                            pl.col("event_idx")
                            .filter(
                                pl.col("time")
                                <=
                                (
                                    pl.col("time")
                                    .filter(
                                        pl.col("code_type") == "ICU_ADMISSION"
                                    )
                                    .first()
                                    +
                                    pl.duration(hours=48)
                                )
                            )
                            .max()
                        )
                        .alias("_w48_time_max"),
                    ]
                )
            )


            # hospital discharge + mortality are admission-level
            hosp_end = (
                df
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                    ]
                )
                .agg(
                    [
                        pl.col("event_idx")
                        .filter(
                            pl.col("code_type") == "HOSPITAL_DISCHARGE"
                        )
                        .first()
                        .alias("hosp_disch_idx"),

                        pl.col("event_idx")
                        .filter(
                            pl.col("code") == "HOSPITAL_DISCHARGE//DIED"
                        )
                        .first()
                        .alias("hospital_mortality_idx"),
                    ]
                )
            )


            icu_windows = (
                icu_window_raw
                .join(
                    hosp_end,
                    on=[
                        "subject_id",
                        "hadm_id",
                    ],
                    how="left",
                )
                .with_columns(
                    [
                        pl.coalesce(
                            [
                                pl.col("hospital_mortality_idx"),
                                pl.col("hosp_disch_idx"),
                            ]
                        )
                        .alias("_clinical_end_idx")
                    ]
                )
                .with_columns(
                    [
                        pl.min_horizontal(
                            [
                                pl.col("_w24_time_max"),
                                pl.col("_clinical_end_idx"),
                            ]
                        )
                        .alias("w24_max"),

                        pl.min_horizontal(
                            [
                                pl.col("_w48_time_max"),
                                pl.col("_clinical_end_idx"),
                            ]
                        )
                        .alias("w48_max"),
                    ]
                )
                .select(
                    [
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                        "w24_max",
                        "w48_max",
                    ]
                )
            )           

            # --------------------------------------------------
            # Hospital mortality
            # --------------------------------------------------
            hospital_mortality = (
                df
                .filter(
                    pl.col("code") == "HOSPITAL_DISCHARGE//DIED"
                )
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                    ]
                )
                .agg(
                    [
                        pl.lit(1).alias("hospital_mortality"),

                        pl.col("time")
                        .first()
                        .alias("hospital_mortality_time"),

                        pl.col("event_idx")
                        .first()
                        .alias("hospital_mortality_idx"),
                    ]
                )
            )


            # --------------------------------------------------
            # MEDS death
            # --------------------------------------------------
            meds_death = (
                df
                .filter(
                    pl.col("code_type") == "MEDS_DEATH"
                )
                .group_by(
                    "subject_id"
                )
                .agg(
                    [
                        pl.lit(1).alias("meds_death"),

                        pl.col("time")
                        .first()
                        .alias("meds_death_time"),

                        pl.col("event_idx")
                        .first()
                        .alias("meds_death_idx"),
                    ]
                )
            )


            # --------------------------------------------------
            # Admission diagnosis/procedure/DRG codes
            # --------------------------------------------------
            admission_codes = (
                df
                .filter(
                    pl.col("hadm_id").is_not_null()
                )
                .group_by(
                    [
                        "subject_id",
                        "hadm_id",
                    ]
                )
                .agg(
                    [
                        pl.struct(
                            [
                                pl.col("code"),
                                pl.col("event_idx"),
                            ]
                        )
                        .filter(
                            pl.col("code_type") == "DIAGNOSIS_ICD"
                        )
                        .alias("icd_events"),

                        pl.struct(
                            [
                                pl.col("code"),
                                pl.col("event_idx"),
                            ]
                        )
                        .filter(
                            pl.col("code_type") == "PROCEDURE_ICD"
                        )
                        .alias("cpt_events"),

                        pl.struct(
                            [
                                pl.col("code"),
                                pl.col("event_idx"),
                            ]
                        )
                        .filter(
                            pl.col("code_type") == "DRG"
                        )
                        .alias("drg_events"),
                    ]
                )
            )


            # --------------------------------------------------
            # Attach metadata only to valid episode rows
            # --------------------------------------------------
            shard_index = (
                stay_index

                .join(
                    hosp_bounds,
                    on=[
                        "subject_id",
                        "hadm_id",
                    ],
                    how="left",
                )

                .join(
                    icu_bounds,
                    on=[
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                    ],
                    how="left",
                )

                .join(
                    icu_windows,
                    on=[
                        "subject_id",
                        "hadm_id",
                        "icustay_id",
                    ],
                    how="left",
                )                

                .join(
                    hospital_mortality,
                    on=[
                        "subject_id",
                        "hadm_id",
                    ],
                    how="left",
                )

                .join(
                    meds_death,
                    on="subject_id",
                    how="left",
                )

                .join(
                    admission_codes,
                    on=[
                        "subject_id",
                        "hadm_id",
                    ],
                    how="left",
                )

                .join(
                    birth_times,
                    on="subject_id",
                    how="left",
                )

                .with_columns(
                    [
                        (
                            (
                                pl.col("hosp_adm_time")
                                -
                                pl.col("birth_time")
                            )
                            .dt.total_days()
                            /
                            365.25
                        )
                        .alias("age_at_admission"),

                        (
                            (
                                pl.col("hosp_disch_time")
                                -
                                pl.col("hosp_adm_time")
                            )
                            .dt.total_hours()
                            /
                            24
                        )
                        .alias("hosp_los"),

                        (
                            (
                                pl.col("icu_disch_time")
                                -
                                pl.col("icu_adm_time")
                            )
                            .dt.total_hours()
                            /
                            24
                        )
                        .alias("icu_los"),

                        pl.col("hospital_mortality")
                        .fill_null(0),

                        pl.col("meds_death")
                        .fill_null(0),

                        pl.lit(shard).alias("shard"),
                        pl.lit(split).alias("split"),
                    ]
                )
            )

            rows.append(shard_index)


    stay_index = pl.concat(
        rows,
        how="vertical"
    )

    return stay_index









def add_query_boundaries(index_df):

    contexts = [512, 1024, 2048]

    # ---------------------------------------------
    # Find earliest forbidden event index
    # ---------------------------------------------
    def get_forbidden_min_idx(row):
        forbidden = []

        for col in [
            "icd_events",
            "cpt_events",
            "drg_events",
        ]:
            events = row[col]

            if events is not None:
                forbidden.extend(
                    [
                        x["event_idx"]
                        for x in events
                    ]
                )

        if row["meds_death_idx"] is not None:
            forbidden.append(
                row["meds_death_idx"]
            )

        if row["hospital_mortality_idx"] is not None:
            forbidden.append(
                row["hospital_mortality_idx"]
            )

        if len(forbidden) == 0:
            return None

        return min(forbidden)


    index_df = index_df.with_columns(
        pl.struct(
            [
                "icd_events",
                "cpt_events",
                "drg_events",
                "meds_death_idx",
                "hospital_mortality_idx",
            ]
        )
        .map_elements(
            get_forbidden_min_idx,
            return_dtype=pl.Int64,
        )
        .alias("_first_forbidden_idx")
    )


    # ---------------------------------------------
    # Safe end boundaries
    # ---------------------------------------------
    index_df = index_df.with_columns(
        [

            pl.when(
                pl.col("_first_forbidden_idx").is_not_null()
            )
            .then(
                pl.min_horizontal(
                    [
                        pl.col("w24_max"),
                        pl.col("_first_forbidden_idx") - 1,
                    ]
                )
            )
            .otherwise(
                pl.col("w24_max")
            )
            .alias("_w24_safe_end"),


            pl.when(
                pl.col("_first_forbidden_idx").is_not_null()
            )
            .then(
                pl.min_horizontal(
                    [
                        pl.col("w48_max"),
                        pl.col("_first_forbidden_idx") - 1,
                    ]
                )
            )
            .otherwise(
                pl.col("w48_max")
            )
            .alias("_w48_safe_end"),


            pl.when(
                pl.col("_first_forbidden_idx").is_not_null()
            )
            .then(
                pl.min_horizontal(
                    [
                        pl.col("wStay_max") - 1,
                        pl.col("_first_forbidden_idx") - 1,
                    ]
                )
            )
            .otherwise(
                pl.col("wStay_max") - 1
            )
            .alias("_wStay_safe_end"),

        ]
    )


    # ---------------------------------------------
    # Generate context boundaries
    # ---------------------------------------------
    expressions = []

    windows = [
        ("w24", "w24_min", "_w24_safe_end"),
        ("w48", "w48_min", "_w48_safe_end"),
        ("wStay", "wStay_min", "_wStay_safe_end"),
    ]


    for prefix, min_col, end_col in windows:

        for context in contexts:

            expressions.extend(
                [
                    pl.col(end_col)
                    .alias(
                        f"{prefix}_end_{context}"
                    ),

                    pl.max_horizontal(
                        [
                            pl.col(min_col),
                            pl.col(end_col)
                            -
                            (context - 1)
                            +
                            1,
                        ]
                    )
                    .alias(
                        f"{prefix}_start_{context}"
                    ),
                ]
            )


    return (
        index_df
        .with_columns(expressions)
        .drop(
            [
                "_first_forbidden_idx",
                "_w24_safe_end",
                "_w48_safe_end",
                "_wStay_safe_end",
            ]
        )
    )





