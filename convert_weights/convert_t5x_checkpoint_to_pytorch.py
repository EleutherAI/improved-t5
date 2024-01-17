# import nest_asyncio
# nest_asyncio.apply()

import abc
import asyncio
from concurrent.futures import thread
import os
import re
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from flax import traverse_util
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorstore as ts

import t5x
import collections
import torch
import transformers

from huggingface_hub import Repository
from transformers.utils import get_full_repo_name

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hf_model', type=str, default="",
                    help='Huggingface Model name')
parser.add_argument('--cache_dir', type=str, default="",
                    help='cache_dir for saving and loading Huggingface Model name')
parser.add_argument('--t5x_ckpt_dir', type=str, default="",
                    help='T5X checkpoint directory')
parser.add_argument('--save_dir', type=str, default="",
                    help='directory for saving the HF model with transferred checkpoint weight')
parser.add_argument('--hub_model_name', type=str, default=None,
                    help='If provided, pushes the model to Huggingface Hub with this name.')
parser.add_argument('--hf_org', type=str, default=None,
                    help='Huggingface Organization name')

args = parser.parse_args()

class LazyArray(abc.ABC):
  """Lazily and asynchronously loads an array."""

  def __init__(self, shape: Sequence[int], dtype: jnp.dtype,
               get_fn: Callable[[], np.ndarray]):
    self._shape = tuple(shape)
    self._dtype = jnp.dtype(dtype)
    self._get_fn = get_fn

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  @property
  def dtype(self) -> jnp.dtype:
    return self._dtype

  @property
  def nbytes(self) -> int:
    return np.prod(self._shape) * self._dtype.itemsize

  def astype(self, dtype: np.dtype) -> 'LazyArray':
    return type(self)(self._shape, dtype, self._get_fn)  # pytype: disable=not-instantiable

  @abc.abstractmethod
  def get_async(self) -> asyncio.Future:
    pass

  @abc.abstractmethod
  def get(self) -> np.ndarray:
    pass

  def __repr__(self):
    return f'{type(self).__name__}(shape={self.shape}, dtype={self.dtype})'

class LazyThreadPoolArray(LazyArray):
  """Lazily and asynchronously loads an array when the `get_fn` blocks."""

  # Uses a global threadpool to enable asynchronous loading.
  executor = thread.ThreadPoolExecutor()

  def get_async(self) -> asyncio.Future:
    return asyncio.wrap_future(self.executor.submit(self.get))

  def get(self) -> np.ndarray:
    arr = self._get_fn()
    if arr.dtype != self.dtype:
      arr = arr.astype(self.dtype)
    return

def load_tf_ckpt(path):
  """Load a TF checkpoint as a flat dictionary of numpy arrays."""
  ckpt_reader = tf.train.load_checkpoint(path)
  ckpt_shape_map = ckpt_reader.get_variable_to_shape_map()
  ckpt_dtype_map = ckpt_reader.get_variable_to_dtype_map()
  datamap = {  # pylint: disable=g-complex-comprehension
      k: LazyThreadPoolArray(
          s,
          jnp.dtype(ckpt_dtype_map[k].as_numpy_dtype),
          lambda x=k: ckpt_reader.get_tensor(x))
      for k, s in ckpt_shape_map.items()
  }
  return datamap

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# checkpoint link: https://console.cloud.google.com/storage/browser/bigscience-t5x/multilingual_t0/mt0_xl_t0pp/checkpoint_1025000;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
# t5x_ckpt_dir = "/Users/zhengxinyong/Desktop/bigscience/mt0_weight_conversion"
init_vars = t5x.checkpoints.load_t5x_checkpoint(args.t5x_ckpt_dir)

print(f"✅ Done loading T5X checkpoint from {args.t5x_ckpt_dir}.")
# print(init_vars)

#### map T5X paramaeters' names to weights
tf_weights = flatten(init_vars, sep="/")
tf_target_weights = {key: weight for key, weight in tf_weights.items() if 'target' in key}

# for name in tf_target_weights.keys():
#     print(name, tf_target_weights[name].shape)

hf_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.hf_model, cache_dir=args.cache_dir)
print(f"✅ Done loading HF model {args.hf_model} from {args.cache_dir}.")

# for name, param in hf_model.named_parameters():
#     print(name, param)
#     assert False


#### use conversion_D dictionary to map HF model parameters' names to T5X paramaeters' name
conversion_D = dict()
for name, param in hf_model.named_parameters():
    if name == "shared.weight":
        conversion_D[name] = "target/token_embedder/embedding"

    elif name.startswith("encoder.block"):
        if "SelfAttention" in name:
            _, _, block_num, _, _, _, key_name, _ = name.split(".")
            if key_name == "relative_attention_bias":
                t5x_layer = "target/encoder/relpos_bias/rel_embedding"
            else:
                key_name = {
                    "q": "query", 
                    "k": "key",
                    "v": "value",
                    "o": "out"
                }[key_name]
                t5x_layer = f"target/encoder/layers_{block_num}/attention/{key_name}/kernel"
            conversion_D[name] = t5x_layer
        elif "DenseReluDense" in name:
            _, _, block_num, _, _, _, key_name, _ = name.split(".")
            t5x_layer = f"target/encoder/layers_{block_num}/mlp/{key_name}/kernel"
            conversion_D[name] = t5x_layer
        elif "layer_norm" in name:
            _, _, block_num, _, layer_num, _,  _ = name.split(".")
            if layer_num == "0":
                t5x_layer = f"target/encoder/layers_{block_num}/pre_attention_layer_norm/scale"
            elif layer_num == "1":
                t5x_layer = f"target/encoder/layers_{block_num}/pre_mlp_layer_norm/scale"
            else:
                assert False
            conversion_D[name] = t5x_layer
        else:
            assert False

    elif name.startswith("encoder.final_layer_norm"):
        conversion_D[name] = "target/encoder/encoder_norm/scale"

    elif name.startswith("lm_head"):
        conversion_D[name] = "target/decoder/logits_dense/kernel"

    elif name.startswith("decoder.block"):
        if "SelfAttention" in name:
            _, _, block_num, _, _, _, key_name, _ = name.split(".")
            if key_name == "relative_attention_bias":
                t5x_layer = "target/decoder/relpos_bias/rel_embedding"
            else:
                key_name = {
                    "q": "query", 
                    "k": "key",
                    "v": "value",
                    "o": "out"
                }[key_name]
                t5x_layer = f"target/decoder/layers_{block_num}/self_attention/{key_name}/kernel"
            conversion_D[name] = t5x_layer
        elif "EncDecAttention" in name:
            _, _, block_num, _, _, _, key_name, _ = name.split(".")
            key_name = {
                "q": "query", 
                "k": "key",
                "v": "value",
                "o": "out"
            }[key_name]
            t5x_layer = f"target/decoder/layers_{block_num}/encoder_decoder_attention/{key_name}/kernel"
            conversion_D[name] = t5x_layer
        elif "DenseReluDense" in name:
            _, _, block_num, _, _, _, key_name, _ = name.split(".")
            t5x_layer = f"target/decoder/layers_{block_num}/mlp/{key_name}/kernel"
            conversion_D[name] = t5x_layer
        elif "layer_norm" in name:
            _, _, block_num, _, layer_num, _,  _ = name.split(".")
            if layer_num == "0":
                t5x_layer = f"target/decoder/layers_{block_num}/pre_self_attention_layer_norm/scale"
            elif layer_num == "1":
                t5x_layer = f"target/decoder/layers_{block_num}/pre_cross_attention_layer_norm/scale"
            elif layer_num == "2":
                t5x_layer = f"target/decoder/layers_{block_num}/pre_mlp_layer_norm/scale"
            else:
                assert False
            conversion_D[name] = t5x_layer
        else:
            assert False

    elif name.startswith("decoder.final_layer_norm"):
        conversion_D[name] = "target/decoder/decoder_norm/scale"

    else:
        assert False
    
    assert conversion_D[name] in tf_target_weights

print(f"✅ Done mapping T5X checkpoint names to HF parameters")

#### replace weights in HF model with T5X checkpoints' weights through conversion_D and tf_target_weights
for name, param in hf_model.named_parameters():
    assert name in conversion_D
    t5x_weight = torch.from_numpy(tf_target_weights[conversion_D[name]])
    if "DenseReluDense" in name or "lm_head" in name or "EncDecAttention" in name or "SelfAttention" in name:
        t5x_weight = t5x_weight.transpose(0, 1)
    
    assert param.data.shape == t5x_weight.shape
    param.data = t5x_weight
    #del conversion_D[name]

#print(f"✅ Done transferring T5X weights to HF model. Remaining weights from T5X (untransferred): {conversion_D}")

if args.hub_model_name is not None:
    repo_name = get_full_repo_name(args.hf_model, organization=args.organization)
    print(f"Creating HuggingFace Hub repository {repo_name}...")
    repo = Repository(args.save_dir, clone_from=repo_name)

    with open(os.path.join(args.save_dir, ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
            gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
            gitignore.write("epoch_*\n")

hf_model.save_pretrained(args.save_dir)
print(f"✅ Done saving to {args.save_dir}.")

tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_model, cache_dir=args.cache_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"✅ Done saving tokenizer to {args.save_dir}.")

if args.hub_model_name is not None:
    repo.push_to_hub()
    print(f"✅ Done pushing to HuggingFace Hub.")

print(f"✅ Done.")
