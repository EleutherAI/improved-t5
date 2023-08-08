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
import functools
import tensorflow_datasets as tfds

from t5.evaluation import metrics
from t5.data import preprocessors
from t5.data import postprocessors
from t5.data.glue_utils import (
    get_glue_postprocess_fn, 
    get_glue_text_preprocessor, 
    get_super_glue_metric, 
    get_super_glue_weight_mapping, 
    get_super_glue_weight_mapping_sentinel
)

from data.sglue.preprocessors import natural_wsc_simple, get_natural_text_preprocessor

from data.vocab import DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== Super GLUE ======================================
# Adapted from https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/tasks.py
# Original T5 SGLUE
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
    # We use a simplified version of WSC, defined below
    if "wsc" in b.name:
        continue
    else:
        glue_preprocessors = [
            get_natural_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ]

    for OUTPUT_FEATURES in [DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES]:
        if OUTPUT_FEATURES == T5_OUTPUT_FEATURES:
            task_name = f"natural_super_glue_{b.name}_v102_t5"
        else:
            task_name = f"natural_super_glue_{b.name}_v102"

        TaskRegistry.add(
            task_name,
            source=seqio.TfdsDataSource(
                tfds_name="super_glue/%s:1.0.2" % b.name,
                splits=["test"] if b.name in ["axb", "axg"] else None),
            preprocessors=glue_preprocessors,
            metric_fns=get_super_glue_metric(b.name),
            output_features=OUTPUT_FEATURES,
            postprocess_fn=get_glue_postprocess_fn(b))


# ======================== Definite Pronoun Resolution =========================
# TaskRegistry.add(
#     "dpr_v001_simple",
#     source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
#     preprocessors=[
#         preprocessors.definite_pronoun_resolution_simple,
#         seqio.preprocessors.tokenize,
#         seqio.CacheDatasetPlaceholder(),
#         seqio.preprocessors.append_eos_after_trim,
#     ],
#     metric_fns=[metrics.accuracy],
#     output_features=DEFAULT_OUTPUT_FEATURES)


# =================================== WSC ======================================
for OUTPUT_FEATURES in [DEFAULT_OUTPUT_FEATURES, T5_OUTPUT_FEATURES]:
    if OUTPUT_FEATURES == T5_OUTPUT_FEATURES:
        train_task_name = "natural_super_glue_wsc_v102_simple_train_t5"
        eval_task_name = "natural_super_glue_wsc_v102_simple_eval_t5"
    else:
        train_task_name = "natural_super_glue_wsc_v102_simple_train"
        eval_task_name = "natural_super_glue_wsc_v102_simple_eval"

    TaskRegistry.add(
        train_task_name,
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
        preprocessors=[
            functools.partial(natural_wsc_simple, correct_referent_only=True),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[],
        output_features=OUTPUT_FEATURES)

    TaskRegistry.add(
        eval_task_name,
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
        preprocessors=[
            functools.partial(
                natural_wsc_simple, correct_referent_only=False),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=postprocessors.wsc_simple,
        metric_fns=[metrics.accuracy],
        output_features=OUTPUT_FEATURES)

# These weights are based on the number of examples in each dataset.
SUPER_GLUE_WEIGHT_MAPPING = {
    # "dpr_v001_simple": 1_322.,
    "natural_super_glue_wsc_v102_simple_train": 259.,
    "natural_super_glue_wsc_v102_simple_eval": 0.,
    "natural_super_glue_boolq_v102": 9_427.,
    "natural_super_glue_cb_v102": 250.,
    "natural_super_glue_copa_v102": 400.,
    "natural_super_glue_multirc_v102": 27_243.,
    "natural_super_glue_record_v102": 138_854.,
    "natural_super_glue_rte_v102": 2_490.,
    "natural_super_glue_wic_v102": 5_428.,
    # "super_glue_axb_v102": 0.,
    # "super_glue_axg_v102": 0.,
}

SUPER_GLUE_WEIGHT_MAPPING_T5 = {k+"_t5": v for k,v in SUPER_GLUE_WEIGHT_MAPPING.items()}

_super_glue_tasks_with_weight = list(SUPER_GLUE_WEIGHT_MAPPING.items())
_super_glue_tasks_with_weight_t5 = list(SUPER_GLUE_WEIGHT_MAPPING_T5.items())

MixtureRegistry.add(
    "natural_super_glue_v102_proportional",
    _super_glue_tasks_with_weight
)

MixtureRegistry.add(
    "natural_super_glue_v102_proportional_t5",
    _super_glue_tasks_with_weight_t5
)