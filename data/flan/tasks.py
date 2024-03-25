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
import os
import datasets
import functools
import seqio
import gcsfs

import tensorflow as tf
import t5.data


from t5.evaluation import metrics

from data.vocab import DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES

from data.utils import CustomDataSource, extract_text_from_jsonl_tf, extract_text_from_json_tf

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

@seqio.map_over_dataset
def extract_text_from_jsonl_tf(json: str):
    inputs = tf.strings.split(json, '"{inputs": "', maxsplit=1)[1]
    inputs = tf.strings.split(inputs, '",', maxsplit=1)[0]

    targets = tf.strings.split(json, '"targets": "', maxsplit=1)[1]
    targets = tf.strings.split(targets, '",', maxsplit=1)[0]

    return {
        "inputs": inputs,
        "targets": targets
        }

for OUTPUT_FEATURES in [DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES]:

    for flan_split in FLAN_SPLIT:

        # flan_task = flan_split.split("/")[-1]
        flan_task = flan_split.split("_data")[0]
        if OUTPUT_FEATURES == T5_OUTPUT_FEATURES:
            task_name = f"{flan_task}_t5"
        else:
            task_name = f"{flan_task}"
        task_name = task_name.replace("-", "_")

        fs = gcsfs.GCSFileSystem()
        file_path = f"gs://improved-t5/flan/{flan_split}"
        file_list = [f"gs://{file}" for file in fs.ls(file_path)], # os.listdir(file_path),
        file_dict = {
            "train": list(file_list),
            "validation": list(file_list[:1]),
            "test": list(file_list[:1]),
            }

        extract_text = extract_text_from_jsonl_tf
        TaskRegistry.add(
            task_name,
            source=CustomDataSource(
                split_to_filepattern=file_dict,
            ),
            preprocessors=[
                extract_text,
                functools.partial(
                    t5.data.preprocessors.rekey, key_map={
                        "inputs": "inputs",
                        "targets": "targets"
                    }),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
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