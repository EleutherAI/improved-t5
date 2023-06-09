# Copyright 2022

"""Defines the vocabulary"""
import seqio

# DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"  # GCS
DEFAULT_SPM_PATH = "gs://improved-t5/vocabs/tokenizer.model"  # Final model tokenizer
# DEFAULT_SPM_PATH = "/fsx/lintangsutawika/improved_t5/tokenizer.model"

def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(
        DEFAULT_SPM_PATH,
        extra_ids=100,
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
