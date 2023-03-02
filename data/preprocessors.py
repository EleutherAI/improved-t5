import gin
import seqio
import tensorflow.compat.v2 as tf

# from t5.data import preprocessors
import t5.data


@gin.configurable()
def identity_tokens(tokens, noise_mask, vocabulary, seeds):
    del noise_mask, vocabulary, seeds
    return tokens

@gin.configurable()
def noise_token_to_mask_token(tokens, noise_mask, vocabulary, seeds):
    """Replace each noise token with mask token.
    Args:
        tokens: a 1d integer Tensor
        noise_mask: a boolean Tensor with the same shape as tokens
        vocabulary: a vocabulary.Vocabulary
        seeds: an unused int32 Tensor
    Returns:
        a Tensor with the same shape and dtype as tokens
    """
    del seeds
    return tf.where(
                noise_mask,
                tf.cast(vocabulary.vocab_size - 1, tokens.dtype),
                tokens
                )

def full_lm(dataset, sequence_length, output_features):
  """Full language modeling objective with EOS only at document boundaries."""
  ds = dataset
  ds = t5.data.preprocessors.select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)
  ds = seqio.preprocessors.append_eos(ds, output_features)
  ds = t5.data.preprocessors.reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  # Don't use `split_tokens_to_targets_length` since we've alrady added EOS.
  ds = t5.data.preprocessors.split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
  return ds

def pack_lm_decoder_only(dataset,
                            sequence_length,
                            loss_on_targets_only=True,
                            pad_id=0):
    """Randomly split the tokens for the prefix LM objective."""
    packed_length = next(iter(sequence_length.values()))
    assert packed_length % 2 == 0
    assert all(l == packed_length for l in sequence_length.values())

    @seqio.utils.map_over_dataset(num_seeds=1)
    def pack_examples(example, seed):
        # split_point = tf.random.stateless_uniform((),
        #                                         minval=1,
        #                                         maxval=packed_length,
        #                                         seed=seed,
        #                                         dtype=tf.int32)
        
        decoder_target_tokens = example['targets']
        decoder_input_tokens = seqio.utils.make_autoregressive_inputs(
            example['inputs']
            )

        # if loss_on_targets_only:
        #     decoder_loss_weights = tf.cast(
        #         tf.range(packed_length) >= split_point, tf.int32)
        # else:
        decoder_loss_weights = tf.ones((packed_length,), dtype=tf.int32)

        padding_mask = tf.cast(
            tf.not_equal(decoder_target_tokens, pad_id), dtype=tf.int32)
        decoder_loss_weights *= padding_mask

        decoder_causal_attention = tf.zeros((packed_length,), tf.int32)

        return {
            'decoder_target_tokens': decoder_target_tokens,
            'decoder_input_tokens': decoder_input_tokens,
            'decoder_causal_attention': decoder_causal_attention,
            'targets': decoder_target_tokens
        }

    return pack_examples(dataset)
