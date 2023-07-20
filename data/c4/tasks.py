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

import t5.data
from data.metrics import perplexity

TaskRegistry = seqio.TaskRegistry

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"  # GCS

def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(
        DEFAULT_SPM_PATH,
        )

DEFAULT_OUTPUT_FEATURES = {
    "inputs": 
        seqio.Feature(
            vocabulary=get_default_vocabulary(),
            add_eos=True,
            required=False),
    "targets":
        seqio.Feature(
            vocabulary=get_default_vocabulary(),
            add_eos=True)
}

DEFAULT_CLM_OUTPUT_FEATURES = {
    "targets":
        seqio.Feature(
            vocabulary=get_default_vocabulary(),
            add_eos=True,
            required=False),
}

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye

# Masked Language Modeling
def make_mlm_task(
    name,
    noise_density=0.15,
    mean_noise_span_length=3.0
    ):
    TaskRegistry.add(
        name,
        source=seqio.TfdsDataSource(tfds_name="c4/en:3.1.0"),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(
                t5.data.preprocessors.span_corruption,
                **{
                    "noise_density": noise_density,
                    "mean_noise_span_length": mean_noise_span_length
                    }
                ),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[perplexity]
    )


# Causal Language Modeling
def make_clm_task(
    name,
    ):
    TaskRegistry.add(
        name,
        source=seqio.TfdsDataSource(tfds_name="c4/en:3.1.0"),
        preprocessors=[
            t5.data.preprocessors.lm,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[perplexity]
    )


# Prefix Language Modeling
def make_plm_task(
    name,
    noise_density=0.5,
    ):
    TaskRegistry.add(
        name,
        source=seqio.TfdsDataSource(tfds_name="c4/en:3.1.0"),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            t5.data.preprocessors.prefix_lm,
            # functools.partial(
            #     t5.data.preprocessors.prefix_lm,
            #     noise_density=noise_density,
            #     ),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[perplexity]
    )

name = 'c4_r_denoiser'
make_mlm_task(name)

name = 'c4_s_denoiser'
make_plm_task(name)

name = 'c4_x_denoiser'
make_mlm_task(name, **{"noise_density": 0.5, "mean_noise_span_length": 32})

name = 'c4_causal_lm'
make_clm_task(name)

seqio.MixtureRegistry.add(
    "c4_ul2",
    ["c4_r_denoiser", "c4_s_denoiser", "c4_x_denoiser"],
    default_rate=1
    )

seqio.MixtureRegistry.add(
    "c4_ul2_causal_0_50",
    [("c4_ul2", 0.50), ("c4_causal_lm", 0.50)],
    )