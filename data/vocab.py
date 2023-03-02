# Copyright 2022

"""Defines the vocabulary"""
import seqio

# # TODO: Link to Eleuther's custom default tokenizer when ready.
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"  # GCS

def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)

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
