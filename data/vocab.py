# Copyright 2022

"""Defines the vocabulary"""
import seqio

T5_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"  # GCS
DEFAULT_SPM_PATH = "gs://improved-t5/vocabs/tokenizer.model"  # LLAMA Tokenizer

def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(
        DEFAULT_SPM_PATH,
        extra_ids=100,
        )

def get_t5_vocabulary():
    return seqio.SentencePieceVocabulary(
        T5_SPM_PATH,
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

T5_OUTPUT_FEATURES = {
    "inputs": 
        seqio.Feature(
            vocabulary=get_t5_vocabulary(),
            add_eos=True,
            required=False),
    "targets":
        seqio.Feature(
            vocabulary=get_t5_vocabulary(),
            add_eos=True)
}
