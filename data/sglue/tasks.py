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

from data.vocab import DEFAULT_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== Super GLUE ======================================
# Original T5 SGLUE
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
    # We use a simplified version of WSC, defined below
    if "wsc" in b.name:
        continue
    if b.name == "axb":
        glue_preprocessors = [
            functools.partial(
                preprocessors.rekey,
                key_map={
                    "premise": "sentence1",
                    "hypothesis": "sentence2",
                    "label": "label",
                    "idx": "idx",
                }),
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ]
    else:
        glue_preprocessors = [
            get_glue_text_preprocessor(b),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ]
    TaskRegistry.add(
        "super_glue_%s_v102" % b.name,
        source=seqio.TfdsDataSource(
            tfds_name="super_glue/%s:1.0.2" % b.name,
            splits=["test"] if b.name in ["axb", "axg"] else None),
        preprocessors=glue_preprocessors,
        metric_fns=get_super_glue_metric(b.name),
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=get_glue_postprocess_fn(b))

    # Create SuperGLUE tasks with 1 sentinel token added.
    seqio.experimental.add_task_with_sentinels(
        "super_glue_%s_v102" % b.name, num_sentinels=1
        )

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple",
    source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
    preprocessors=[
        preprocessors.definite_pronoun_resolution_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("dpr_v001_simple", num_sentinels=1)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_train",
                                           num_sentinels=1)

TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            preprocessors.wsc_simple, correct_referent_only=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)
# Create SuperGLUE tasks with 1 sentinel token added.
seqio.experimental.add_task_with_sentinels("super_glue_wsc_v102_simple_eval",
                                           num_sentinels=1)

_SUPER_GLUE_WEIGHT_MAPPING = get_super_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING_SENTINEL = get_super_glue_weight_mapping_sentinel()

_super_glue_tasks_with_weight = list(_SUPER_GLUE_WEIGHT_MAPPING.items())
_super_glue_tasks_with_weight_sentinel = list(
    _SUPER_GLUE_WEIGHT_MAPPING_SENTINEL.items())

MixtureRegistry.add(
    "super_glue_v102_proportional",
    _super_glue_tasks_with_weight
)

MixtureRegistry.add(
    "super_glue_v102_proportional_sentinel",
    _super_glue_tasks_with_weight_sentinel
)