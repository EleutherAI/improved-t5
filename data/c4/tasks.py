"""
To cache tasks before training,

seqio_cache_tasks \
    --tasks=my_task_*,your_task \
    --excluded_tasks=my_task_5 \
    --output_cache_dir=/path/to/cache_dir \
    --module_import=my.tasks \
    --alsologtostderr

For more details, see: seqio/scripts/cache_tasks_main.py

"""

import seqio

from data.c4 import c4_utils
from data.utils import make_mlm_task, make_clm_task #, make_fcm_task

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye

path="/fsx/c4/c4-en"
c4_files = c4_utils.get_c4_files(path)

# Masked Language Model Format
make_mlm_task('c4_eye_span_corruption', c4_files, jsonl=False)
make_mlm_task('c4_mlm_0_10', c4_files, jsonl=False, noise_density=0.10)
make_mlm_task('c4_mlm_0_15', c4_files, jsonl=False, noise_density=0.15)
make_mlm_task('c4_mlm_0_25', c4_files, jsonl=False, noise_density=0.25)
make_mlm_task('c4_mlm_0_50', c4_files, jsonl=False, noise_density=0.50)
make_mlm_task('c4_mlm_0_75', c4_files, jsonl=False, noise_density=0.75)
make_mlm_task('c4_mlm_1_00', c4_files, jsonl=False, noise_density=1.00)

# Causal Language Model Format
make_clm_task('c4_eye_full_lm', c4_files, jsonl=False)

# # Forgetful Causal Masking (https://arxiv.org/abs/2210.13432)
# make_fcm_task('c4_eye_fcm_full_lm', c4_files, jsonl=False)


