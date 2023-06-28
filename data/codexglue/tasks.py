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
from functools import partial
from data.vocab import DEFAULT_CLM_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== CodeXGLUE Code to Text ======================================

# path="gs://improved-t5/code-x-glue/dedupe0.87/train/"
CODE_LANG = ['go', 'java', 'javascript', 'php', 'python', 'ruby']

def dataset_fn(split, shuffle_files, code_lang):

    ds = datasets.load_dataset(f"CM/codexglue_code2text_{code_lang}")
    ds = ds[split]
    ds = datasets.Dataset.from_dict({
        "code_tokens": ds["code_tokens"],
        "docstring_tokens": ds["docstring_tokens"],
        })
    ds = ds.with_format("tf")
    return ds


@seqio.map_over_dataset
def code_to_text_preprocessor(x):

    inputs = tf.strings.regex_replace(
        tf.strings.join(x['code_tokens'], separator=' '), '\n', ' '
        )
    inputs = tf.strings.join(
        tf.strings.split(
            tf.strings.strip(inputs)
            ),
        separator=' '
        )

    targets = tf.strings.regex_replace(
        tf.strings.join(x['docstring_tokens'], separator=' '), '\n', ''
        )
    targets = tf.strings.join(
        tf.strings.split(
            tf.strings.strip(targets)
            ),
        separator=' '
        )

    return {
        'inputs': inputs,
        'targets': targets
    }

for code_lang in CODE_LANG:
    TaskRegistry.add(
        "code-to-text-{}".format(code_lang),
        source=seqio.FunctionDataSource(
            dataset_fn=partial(dataset_fn, code_lang=code_lang),
            splits=["train", "validation", "test"]
        ),
        preprocessors=[
            code_to_text_preprocessor,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[],
        output_features=DEFAULT_OUTPUT_FEATURES
    )


seqio.MixtureRegistry.add(
    "code_x_glue_code_to_text",
    [f"code-to-text-{code_lang}" for code_lang in CODE_LANG],
    default_rate=1
    )

