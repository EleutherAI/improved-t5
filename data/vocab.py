# Copyright 2022

"""Defines the vocabulary"""
import seqio
import numpy as np
import tensorflow as tf

from transformers import AutoTokenizer

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

# class AltSentencePieceVocabulary(seqio.SentencePieceVocabulary):
#     def _encode_tf(self, s):
#         print(s)
#         print(self.tf_tokenizer.tokenize(s))
#         return self.tf_tokenizer.tokenize(s)

# def get_default_vocabulary():
#     return AltSentencePieceVocabulary(DEFAULT_SPM_PATH)

# DEFAULT_OUTPUT_FEATURES = {
#     "inputs": 
#         seqio.Feature(
#             vocabulary=get_default_vocabulary(),
#             add_eos=True,
#             required=False),
#     "targets":
#         seqio.Feature(
#             vocabulary=get_default_vocabulary(),
#             add_eos=True)
# }



# class HFVocabulary(seqio.Vocabulary):
#     def __init__(
#         self,
#         pretrained_model_name_or_path
#     ):
        
#         self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

#     @property
#     def eos_id(self):
#         return self._tokenizer.eos_token_id

#     @property
#     def pad_id(self):
#         return self._tokenizer.pad_token_id

#     @property
#     def unk_id(self):
#         return self._tokenizer.unk_token_id

#     @property
#     def vocab_size(self):
#         return self._base_vocab_size

#     @property
#     def _base_vocab_size(self):
#         return self._tokenizer.vocab_size

#     def _encode(self, s):
#         return self._tokenizer.encode(s)

#     def _decode(self, ids):
#         return self._tokenizer.decode(ids)

#     @tf.function
#     def _tokenizer(self, s):
#         # s = bytes.decode(s.numpy())
#         s = s.numpy()
#         return self._tokenizer.encode(s, return_tensors='tf')[0]

#     def _encode_tf(self, s):
#         print(s)
#         # s = bytes.decode(s.numpy())
#         # s = bytes.decode(s)
#         return self._tokenizer(s)
#         # s = tf.get_static_value(s)
#         # if s == None:
#         #     s = ""
#         # s = self._tokenizer.encode(s, return_tensors='tf')[0]
#         # s = tf.cast(s, tf.int32)
#         return s 

#     def _decode_tf(self, ids):
#         s = self._tokenizer.decode(ids)
#         return tf.convert_to_tensor(s, dtype=tf.string)

# def get_hf_vocabulary():
#     return HFVocabulary("bigscience/bloom")


# DEFAULT_OUTPUT_FEATURES = {
#     "inputs": 
#         seqio.Feature(
#             vocabulary=get_hf_vocabulary(),
#             add_eos=True,
#             required=False),
#     "targets":
#         seqio.Feature(
#             vocabulary=get_hf_vocabulary(),
#             add_eos=True)
# }
