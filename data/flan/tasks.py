"""
To cache tasks before training,

seqio_cache_tasks \
    --tasks=my_task_*,your_task \
    --excluded_tasks=my_task_5 \
    --output_cache_dir=/path/to/cache_dir \
    --module_import=my.tasks \
    --alsologtostderr

For more details, see: seqio/scripts/cache_tasks_main.py

"""

import seqio
import json
import os
import datasets

import tensorflow as tf

from t5.evaluation import metrics

from functools import partial
from data.vocab import DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== FLAN ======================================

FLAN_SPLIT = [
    "niv2_zsopt_data",
    "niv2_fsopt_data",
    "flan_zsopt_data",
    "flan_fsopt_data",
    "dialog_zsopt_data",
    "dialog_fsopt_data",
    "t0_zsopt_data",
    "t0_zsnoopt_data",
    "t0_fsopt_data",
    "t0_fsnoopt_data",
    "flan_zsnoopt_data",
    "flan_fsnoopt_data",
    "cot_fsopt_data",
    "cot_zsopt_data",
    ]

def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(shape=() if not length else (None if length == -1 else length,), dtype=tf.int64)
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=getattr(tf.dtypes, feature.dtype)
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def flan_preprocessor(x):

    return {
        "inputs": x["inputs"],
        "targets": x["targets"]
    }


def dataset_fn(split, shuffle_files, seed=None, dataset=None):

    # ds = datasets.load_dataset(
    #     "Open-Orca/FLAN",
    #     num_proc = os.cpu_count(),
    #     data_files=f"FLAN/{dataset}/*.parquet"
    #     )

    ds = load_dataset(
        "parquet",
        num_proc=os.cpu_count(),
        data_files=f"gs://improved-t5/FLAN/{dataset}/*.parquet")

    ds = ds[split]

    ds = ds.map(flan_preprocessor)
    return tf.data.Dataset.from_generator(
            ds.__iter__, output_signature={k: feature_to_spec(v) for k, v in ds.features.items()}
        )


for OUTPUT_FEATURES in [DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES]:

    for flan_split in FLAN_SPLIT:

        # flan_task = flan_split.split("/")[-1]
        flan_task = flan_split.split("_data")[0]
        if OUTPUT_FEATURES == T5_OUTPUT_FEATURES:
            task_name = f"{flan_task}_t5"
        else:
            task_name = f"{flan_task}"
        task_name = task_name.replace("-", "_")

        TaskRegistry.add(
            task_name,
            source=seqio.FunctionDataSource(
                dataset_fn=partial(dataset_fn, dataset=flan_split),
                splits=["train"]
            ),
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            # metric_fns=[metrics.accuracy],
            metric_fns=[],
            output_features=OUTPUT_FEATURES
        )

seqio.MixtureRegistry.add(
    'cot_submix',
    tasks=[
        ('cot_zsopt', 1),    # mixing weight = 50%
        ('cot_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'dialog_submix',
    tasks=[
        ('dialog_zsopt', 1),    # mixing weight = 50%
        ('dialog_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'niv2_submix',
    tasks=[
        ('niv2_zsopt', 1),    # mixing weight = 50%
        ('niv2_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix',
    tasks=[
        ('flan_zsopt', 1),      # mixing weight = 25%
        ('flan_fsopt', 1),      # mixing weight = 25%
        ('flan_zsnoopt', 1),    # mixing weight = 25%
        ('flan_fsnoopt', 1),    # mixing weight = 25%
    ])

seqio.MixtureRegistry.add(
    't0_submix',
    tasks=[
        ('t0_zsopt', 1),      # mixing weight = 25%
        ('t0_fsopt', 1),      # mixing weight = 25%
        ('t0_zsnoopt', 1),    # mixing weight = 25%
        ('t0_fsnoopt', 1),    # mixing weight = 25%
    ])

# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix', 0.4),  # mixing weight = 40%
        ('t0_submix', 0.32),       # mixing weight = 32%
        ('niv2_submix', 0.2),      # mixing weight = 20%
        ('cot_submix', 0.05),      # mixing weight = 5%
        ('dialog_submix', 0.03),   # mixing weight = 3%
    ])

seqio.MixtureRegistry.add(
    'cot_submix_t5',
    tasks=[
        ('cot_zsopt_t5', 1),    # mixing weight = 50%
        ('cot_fsopt_t5', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'dialog_submix_t5',
    tasks=[
        ('dialog_zsopt_t5', 1),    # mixing weight = 50%
        ('dialog_fsopt_t5', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'niv2_submix_t5',
    tasks=[
        ('niv2_zsopt_t5', 1),    # mixing weight = 50%
        ('niv2_fsopt_t5', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix_t5',
    tasks=[
        ('flan_zsopt_t5', 1),      # mixing weight = 25%
        ('flan_fsopt_t5', 1),      # mixing weight = 25%
        ('flan_zsnoopt_t5', 1),    # mixing weight = 25%
        ('flan_fsnoopt_t5', 1),    # mixing weight = 25%
    ])

seqio.MixtureRegistry.add(
    't0_submix_t5',
    tasks=[
        ('t0_zsopt_t5', 1),      # mixing weight = 25%
        ('t0_fsopt_t5', 1),      # mixing weight = 25%
        ('t0_zsnoopt_t5', 1),    # mixing weight = 25%
        ('t0_fsnoopt_t5', 1),    # mixing weight = 25%
    ])

# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix_t5',
    tasks=[
        ('flan2021_submix_t5', 0.4),  # mixing weight = 40%
        ('t0_submix_t5', 0.32),       # mixing weight = 32%
        ('niv2_submix_t5', 0.2),      # mixing weight = 20%
        ('cot_submix_t5', 0.05),      # mixing weight = 5%
        ('dialog_submix_t5', 0.03),   # mixing weight = 3%
    ])