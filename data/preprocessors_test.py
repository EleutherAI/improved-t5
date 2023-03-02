import functools

import gin
import seqio
import random
import tensorflow.compat.v2 as tf

from seqio import test_utils
from absl.testing import absltest

from t5x.data import preprocessors

class preprocessorsTest(tf.test.TestCase):

    def test_masked_language_modeling(self):

        vocab = test_utils.sentencepiece_vocab()
        # inp = [random.randint(0, vocab.vocab_size -2) for i in range(100)]
        inp = list(range(0,100))
        og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [inp]})
        og_dataset = og_dataset.repeat(100)
        output_features = {
            'targets': seqio.Feature(vocab),
            'inputs': seqio.Feature(vocab),
        }

        output_dataset = preprocessors.masked_language_modeling(
            og_dataset,
            sequence_length={'targets': 100, 'inputs': 100},
            output_features=output_features,
            )
        
        output_sample = list(output_dataset.as_numpy_iterator())[0]
        output_keys = output_sample.keys()
        for key in output_keys:
            print("#### {} ###".format(key))
            print(output_sample[key])

        self.assertSequenceEqual(['inputs', 'targets'], list(output_keys))

    def test_pack_prefix_lm_decoder_only(self):

        vocab = test_utils.sentencepiece_vocab()
        inp = [random.randint(0, vocab.vocab_size -2) for i in range(100)]
        og_dataset = tf.data.Dataset.from_tensor_slices({'targets': [inp]})
        og_dataset = og_dataset.repeat(100)
        output_features = {
            'decoder_target_tokens': seqio.Feature(vocab),
            'decoder_input_tokens': seqio.Feature(vocab),
            'decoder_loss_weights': seqio.Feature(vocab),
            'decoder_causal_attention': seqio.Feature(vocab),
        }

        output_dataset = preprocessors.masked_language_modeling(
        # output_dataset = preprocessors.targets_for_prefix_lm_objective(
            og_dataset,
            sequence_length={
                    'targets': 100,
                    'inputs': 100
                },
            output_features={
                    'targets': seqio.Feature(vocab),
                    'inputs': seqio.Feature(vocab),
                },
            )

        output_dataset = seqio.preprocessors.append_eos_after_trim(
            output_dataset,
            sequence_length={
                    'targets': 100,
                    'inputs': 100
                },
            output_features={
                    'targets': seqio.Feature(vocab),
                    'inputs': seqio.Feature(vocab),
                },
            )

        output_dataset = preprocessors.pack_lm_decoder_only(
            output_dataset,
            sequence_length={
                    'targets': 100,
                    'inputs': 100
                },
            # sequence_length={
            #     'decoder_target_tokens': 100,
            #     'decoder_input_tokens': 100,
            #     'decoder_loss_weights': 100,
            #     'decoder_causal_attention': 100,
            # },
            # output_features={
            #     'decoder_target_tokens': seqio.Feature(vocab),
            #     'decoder_input_tokens': seqio.Feature(vocab),
            #     'decoder_loss_weights': seqio.Feature(vocab),
            #     'decoder_causal_attention': seqio.Feature(vocab),
            # },
        )

        # output_dataset = og_dataset
        # for _p in _process:
        #     output_dataset = _p(
        #         output_dataset, 
        #         sequence_length={key: 100 for key in output_features.keys()},
        #         output_features=output_features
        #         )

        # # output_dataset = preprocessors.masked_language_modeling(
        # #     og_dataset,
        # #     sequence_length={'targets': 100, 'inputs': 100},
        # #     output_features=output_features,
        # #     )

        # # output_dataset = preprocessors.pack_prefix_lm_decoder_only(
        # #     output_dataset,
        # #     sequence_length={'targets': 100, 'inputs': 100},
        # #     # output_features=output_features,
        # #     )

        output_sample = list(output_dataset.as_numpy_iterator())[0]
        output_keys = output_sample.keys()
        for key in output_keys:
            print("#### {} ###".format(key))
            print(output_sample[key])

        self.assertSequenceEqual(
            ['decoder_target_tokens',
            'decoder_input_tokens',
            'decoder_loss_weights',
            'decoder_causal_attention'],
            list(output_keys)
        )

    # def test_masked_language_modeling_passthrough(self):
    #     # No merging of examples, passthrough keys
    #     vocab = test_utils.sentencepiece_vocab()
    #     inp = list(range(1, 100))
    #     pt = list(range(1, 20))
    #     og_dataset = tf.data.Dataset.from_tensor_slices({
    #         'targets': [inp],
    #         'passthrough': [pt],
    #     })
    #     og_dataset = og_dataset.repeat(100)
    #     output_features = {
    #         'targets': seqio.Feature(vocab),
    #         'inputs': seqio.Feature(vocab),
    #         'passthrough': seqio.Feature(vocab),
    #     }

    #     output_dataset = preprocessors.masked_language_modeling(
    #         og_dataset,
    #         sequence_length={'targets': 100, 'inputs': 100},
    #         output_features=output_features,
    #         merge_examples_to_reduce_padding=False,
    #         passthrough_feature_keys=['passthrough'])

    #     for ex in output_dataset.as_numpy_iterator():
    #         self.assertLessEqual(len(ex['inputs']), len(inp))
    #         self.assertAllEqual(pt, ex['passthrough'])

    # def test_masked_language_modeling_passthrough_fail(self):
    #     og_dataset = tf.data.Dataset.from_tensor_slices({
    #         'targets': [list(range(1, 100))],
    #         'passthrough': [list(range(1, 20))],
    #     })
    #     with self.assertRaises(ValueError):
    #         _ = preprocessors.masked_language_modeling(
    #             og_dataset,
    #             sequence_length={'targets': 100, 'inputs': 100},
    #             output_features=None,
    #             merge_examples_to_reduce_padding=True,
    #             passthrough_feature_keys=['passthrough'])

if __name__ == '__main__':
    absltest.main()