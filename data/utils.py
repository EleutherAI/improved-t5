# Copyright 2022

"""Utilities for data loading and processing."""

import functools
import os
import gin
import seqio

import tensorflow as tf
from typing import Iterable, Mapping, Optional, Union

import t5.data
from data.metrics import perplexity
from data.vocab import DEFAULT_OUTPUT_FEATURES, DEFAULT_CLM_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry

# @seqio.utils.map_over_dataset(num_seeds=1)
@seqio.map_over_dataset
def extract_text_from_json_tf(json: str):
    output = tf.strings.split(json, '{"text":"', maxsplit=1)[1]
    output = tf.strings.split(output, '",', maxsplit=1)[0]
    return {"text": output}

@seqio.map_over_dataset
def extract_text_from_jsonl_tf(json: str):
    output = tf.strings.split(json, '{"text": "', maxsplit=1)[1]
    output = tf.strings.split(output, '",', maxsplit=1)[0]
    return {"text": output}

# Masked Language Modeling
def make_mlm_task(
    name,
    split_to_filepattern,
    jsonl,
    noise_density=0.15,
    mean_noise_span_length=3.0
    ):
    extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
    TaskRegistry.add(
        name,
        source=CustomDataSource(
            split_to_filepattern=split_to_filepattern,
        ),
        preprocessors=[
            extract_text,
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
                    "mean_noise_span_length": mean_noise_span_length,
                }),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[perplexity]
    )


# Causal Language Modeling
def make_clm_task(
    name,
    split_to_filepattern,
    jsonl
    ):
    extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
    TaskRegistry.add(
        name,
        source=CustomDataSource(
            split_to_filepattern=split_to_filepattern,
        ),
        preprocessors=[
            extract_text,
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
    split_to_filepattern,
    jsonl,
    noise_density=0.5,
    ):
    extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
    TaskRegistry.add(
        name,
        source=CustomDataSource(
            split_to_filepattern=split_to_filepattern,
        ),
        preprocessors=[
            extract_text,
            functools.partial(
                t5.data.preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            t5.data.preprocessors.prefix_lm,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[perplexity]
    )


# def make_fcm_task(name, split_to_filepattern, jsonl):
#     extract_text = extract_text_from_jsonl_tf if jsonl else extract_text_from_json_tf
#     TaskRegistry.add(
#         name,
#         source=CustomDataSource(
#             split_to_filepattern=split_to_filepattern
#         ),
#         preprocessors=[
#             extract_text,
#             functools.partial(
#                 seqio.preprocessors.rekey, key_map={
#                     "inputs": None,
#                     "targets": "text"
#                 }),
#             seqio.preprocessors.tokenize,
#             seqio.CacheDatasetPlaceholder(),
#             preprocessors.masked_language_modeling,
#             seqio.preprocessors.append_eos_after_trim,
#             preprocessors.pack_lm_decoder_only,
#         ],
#         output_features={
#             "decoder_target_tokens": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=False),
#             "decoder_input_tokens": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=False),
#             "decoder_causal_attention": seqio.Feature(vocabulary=seqio.PassThroughVocabulary(1), add_eos=False),
#             # All but the last stage of the preprocessing uses "targets" as the key,
#             # so this output feature is necessary. It is not marked required because
#             # the final preprocessor drops it.
#             "targets": seqio.Feature(vocabulary=get_default_vocabulary(), required=False),
#         },
#         metric_fns=[]
#     )


class CustomDataSource(seqio.FileDataSource):
    """A `FileDataSource` that reads lines of text from a file as input and takes in _TFDS_DATA_DIR_OVERRIDE"""

    def __init__(self,
                split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
                skip_header_lines: int = 0,
                num_input_examples: Optional[Mapping[str, int]] = None,
                caching_permitted: bool = True,
                file_shuffle_buffer_size: Optional[int] = None):
        """TextLineDataSource constructor.
        Args:
        split_to_filepattern: a mapping from split names to filepatterns to be
            expanded with glob.
        skip_header_lines: int, number of header lines to skip in each source
            file.
        num_input_examples: dict or None, an optional dictionary mapping split to
            its size in number of input examples (before preprocessing). The
            `num_input_examples` method will return None if not provided.
        caching_permitted: indicates whether this data source may be cached.
            Default True.
        file_shuffle_buffer_size: The buffer size to shuffle files when needed. If
            None, the number of files is used as buffer size for a perfect shuffle
            (default and recommended). A value of 16 may be explicitly set to
            replicate earlier behavior.
        """
        # Used during caching.
        self._data_dir = seqio.utils._TFDS_DATA_DIR_OVERRIDE
        if self._data_dir is None:
            self._data_dir = ""
        self._skip_header_lines = skip_header_lines

        def read_file_fn(filepattern):
            return tf.data.TextLineDataset(filepattern).skip(skip_header_lines)


        super().__init__(
            read_file_fn=read_file_fn,
            split_to_filepattern={
                k: [os.path.join(self._data_dir, _v) for _v in v] \
                        if type(v) == list else os.path.join(self._data_dir, v) \
                    for k,v in split_to_filepattern.items()
                    },
            num_input_examples=num_input_examples,
            caching_permitted=caching_permitted,
            file_shuffle_buffer_size=file_shuffle_buffer_size)

