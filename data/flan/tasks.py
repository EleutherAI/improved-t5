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
import datasets

import tensorflow as tf

from t5.evaluation import metrics

from functools import partial
from data.vocab import DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== FLAN ======================================

FLAN_SPLIT = [
    "DataProvenanceInitiative/flan2021_submix_original",
    "DataProvenanceInitiative/t0_submix_original",
    "DataProvenanceInitiative/niv2_submix_original",
    "DataProvenanceInitiative/cot_submix_original",
    "DataProvenanceInitiative/dialog_submix_original",
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

    ds = datasets.load_dataset(dataset)
    ds = ds[split]

    ds = ds.map(flan_preprocessor)
    return tf.data.Dataset.from_generator(
            ds.__iter__, output_signature={k: feature_to_spec(v) for k, v in ds.features.items()}
        )


for OUTPUT_FEATURES in [DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES]:

    for flan_split in FLAN_SPLIT:

        flan_task = flan_split.split("/")[-1]
        if OUTPUT_FEATURES == T5_OUTPUT_FEATURES:
            task_name = f"{flan_task}_t5"
        else:
            task_name = f"{flan_task}"

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

# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix_original', 0.4),  # mixing weight = 40%
        ('t0_submix_original', 0.32),       # mixing weight = 32%
        ('niv2_submix_original', 0.2),      # mixing weight = 20%
        ('cot_submix_original', 0.05),      # mixing weight = 5%
        ('dialog_submix_original', 0.03),   # mixing weight = 3%
    ])

seqio.MixtureRegistry.add(
    'flan2022_submix_t5',
    tasks=[
        ('flan2021_submix_original_t5', 0.4),  # mixing weight = 40%
        ('t0_submix_original_t5', 0.32),       # mixing weight = 32%
        ('niv2_submix_original_t5', 0.2),      # mixing weight = 20%
        ('cot_submix_original_t5', 0.05),      # mixing weight = 5%
        ('dialog_submix_original_t5', 0.03),   # mixing weight = 3%
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix',
    tasks=[
        ('flan2021_submix_original', 0.46),
        ('t0_submix_original', 0.28),
        ('niv2_submix_original', 0.242),
        ('cot_submix_original', 0.018),
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix_t5',
    tasks=[
        ('flan2021_submix_original_t5', 0.46),
        ('t0_submix_original_t5', 0.28),
        ('niv2_submix_original_t5', 0.242),
        ('cot_submix_original_t5', 0.018),
    ])