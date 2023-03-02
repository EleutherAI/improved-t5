import seqio
from bigbench.bbseqio import task_api
from bigbench.bbseqio import tasks

from t5x.data.vocab import DEFAULT_OUTPUT_FEATURES, get_default_vocabulary

default_vocab = task_api.SeqIOVocabulary(
    name="default",
    description="default vocab",
    vocabulary=get_default_vocabulary())

# Register BIG-bench lite tasks.
# bigbench:bigbench_lite_v1.mix.default_vocab.0_shot.all_examples
num_shots = 0
tasks.register_bigbench_lite(num_shots, default_vocab)