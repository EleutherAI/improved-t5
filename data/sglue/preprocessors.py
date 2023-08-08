import collections
import functools

import gin
import seqio
import tensorflow.compat.v2 as tf

from t5.data import preprocessors

# pylint:disable=no-value-for-parameter
AUTOTUNE = tf.data.experimental.AUTOTUNE

def natural_wsc_simple(dataset,
               label='wsc:',
               correct_referent_only=False):

    def map_fn(x):
        """Function to be called for every example in dataset."""
        passage = x['text']
        referent = x['span1_text']
        pronoun = x['span2_text']
        inputs = [
            f'Passage: {passage}',
            f'Question: In the passage above, what does the pronoun "{pronoun}" refer to? Answer: '
        ]
        return {
            'inputs': tf.strings.join(inputs, separator='\n'),
            # The reshape is necessary as otherwise the tensor has unknown rank.
            'targets': tf.reshape(referent, shape=[]),
            'label': x.get('label', 0),
            'idx': x['idx'],
        }

    if correct_referent_only:
        dataset = dataset.filter(lambda x: tf.cast(x.get('label', False), tf.bool))

    return dataset.map(map_fn, num_parallel_calls=AUTOTUNE)

def get_natural_text_preprocessor(builder_config):
    """Return the glue preprocessor.

    Args:
        builder_config: a BuilderConfig
    Returns:
        a preprocessor function
    """
    if builder_config.name == "record":
        return record
    else:
        feature_names = None
        label_names = ["No", "Yes"]
        benchmark_name = builder_config.name
        if benchmark_name == "cb":
            label_names = ["No", "Yes", "Maybe"]
        elif benchmark_name == "copa":
            label_names = builder_config.label_classes
        elif benchmark_name == "multirc":
            feature_names = ("question", "answer", "paragraph")
        elif benchmark_name == "wic":
            feature_names = ("sentence1", "sentence2", "word")
        return functools.partial(
            natural_super_glue,
            benchmark_name=benchmark_name,
            label_names=builder_config.label_classes,
            feature_names=feature_names)

@gin.configurable
def record(dataset):

    def process_answers(x):
        """Helper fn to get one example per answer."""
        ex = x.copy()
        num_answers = tf.size(ex['answers'])

        def duplicate_along_first_dim(t):
            n_duplicates = tf.math.maximum(num_answers, 1)
            return tf.broadcast_to(
                t, shape=tf.concat([[n_duplicates], tf.shape(t)], axis=0))

            for k, v in x.items():
                if k != 'idx':
                    ex[k] = duplicate_along_first_dim(v)
            ex['targets'] = tf.cond(
                tf.greater(num_answers, 0), lambda: x['answers'],
                lambda: tf.constant(['<unk>']))
            ex['idx'] = {
                'passage': duplicate_along_first_dim(x['idx']['passage']),
                'query': duplicate_along_first_dim(x['idx']['query']),
            }

        return ex

    def my_fn(x):
        """Converts the processed example to text2text strings."""
        passage = x['passage']
        passage = tf.strings.regex_replace(passage,
                                        r'(\.|\?|\!|\"|\')\n@highlight\n',
                                        r'\1 ')
        passage = tf.strings.regex_replace(passage, r'\n@highlight\n', '. ')
        entities = tf.strings.reduce_join(x['entities'], separator=', ')
        query = x['query']
        strs_to_join = [
            tf.strings.join(["Passage:", passage], separator=' '),
            tf.strings.join(["Entities:", entities], separator=' '),
            tf.strings.join(["Query:", query], separator=' '),
            f'Question: What is the most likely entity to fill in @placeholder in previous query?'
            'Answer: '
        ]
        joined = tf.strings.join(strs_to_join, separator='\n')

        ex = {}

        # Store the data index in the returned example (used by eval)
        ex['idx/passage'] = x['idx']['passage']
        ex['idx/query'] = x['idx']['query']

        ex['inputs'] = joined
        # Note that "answers" has been converted to a single string by the
        # process_answers function.
        ex['targets'] = x['targets']
        # Pass-through full list of answers for eval
        ex['answers'] = x['answers']
        return ex

    dataset = dataset.map(process_answers, num_parallel_calls=AUTOTUNE)
    dataset = dataset.unbatch()
    return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


@seqio.map_over_dataset
def natural_super_glue(x, benchmark_name, label_names, feature_names=None, id_key='idx'):

    # If an ordering is not provided, sort feature keys to ensure a consistent
    # order.
    feature_keys = (
        feature_names or sorted(set(x.keys()).difference(['label', 'idx'])))

    label_name = tf.cond(
        # When no label is provided (label == -1), use "<unk>"
        tf.equal(x['label'], -1),
        lambda: tf.constant('<unk>'),
        # Otherwise grab the label text from label_names
        lambda: tf.gather(label_names, x['label']),
    )

    ex = {}

    if benchmark_name == 'multirc':

        strs_to_join = [
            tf.strings.join(["Passage: ", x['paragraph']], separator=''),
            tf.strings.join(["Query:  ", x['question']], separator=''),
            tf.strings.join(["Question: Is \"", x['answer'],"\" the correct answer to the query given the passage?"], separator=''),
            "Answer: "
            ]

        # Store the data index in the returned example (used by eval)
        ex['idx/paragraph'] = x['idx']['paragraph']
        ex['idx/question'] = x['idx']['question']
        ex['idx/answer'] = x['idx']['answer']
    else:
        # Store the data index in the returned example (used by eval)
        if id_key:
            ex['idx'] = x[id_key]

        if benchmark_name == "boolq":
            strs_to_join = [
                tf.strings.join(["Passage: ", x['passage']], separator=''),
                tf.strings.join(["Question: ", x['question'], "?"], separator=''),
                "Answer: "
                ]
        elif benchmark_name == "cb":
            strs_to_join = [
                tf.strings.join(["Passage: ", x['premise']], separator=''),
                tf.strings.join(["Question: ", x['hypothesis'], "?"], separator=''),
                "Answer: "
                ]
        elif benchmark_name == "copa":
            strs_to_join = [
                tf.strings.join(["Premise:", x['premise']], separator=' '),
                tf.strings.join(["Choice1:", x['choice1']], separator=' '),
                tf.strings.join(["Choice2:", x['choice2']], separator=' '),
                tf.strings.join(["Question: Between choice1 and choice2, which serves as", x['question'], "to the premise?"], separator=' '),
                "Answer: "
                ]
        elif benchmark_name == "rte":
            strs_to_join = [
                tf.strings.join(["Passage: ", x['premise']], separator=''),
                tf.strings.join(["Question: ", x['hypothesis'], "?"], separator=''),
                "Answer: "
                ]
        elif benchmark_name == "wic":
            strs_to_join = [
                tf.strings.join(["Sentence 1:", x['sentence1']], separator=' '),
                tf.strings.join(["Sentence 2:", x['sentence2']], separator=' '),
                tf.strings.join(["Question: Is the word '", x['word'], "' used in the same way for both sentences above?"], separator=' '),
                "Answer: "
                ]

    joined = tf.strings.join(
        strs_to_join,
        separator='\n')

    if benchmark_name == 'multirc':
        # Remove HTML markup.
        joined = tf.strings.regex_replace(joined, '<br>', ' ')
        joined = tf.strings.regex_replace(joined, '<(/)?b>', '')

    ex['inputs'] = joined
    ex['targets'] = label_name

    return ex