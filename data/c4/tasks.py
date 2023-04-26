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
from data.utils import make_mlm_task, make_clm_task, make_plm_task

# ==================================== C4 ======================================
# A version of c4 corresponding to one hosted on the-eye

path="/fsx/c4/c4-en"

def c4_helper(task, name, c4_files, **kwargs):
    task(name, c4_files, jsonl=False, **kwargs)

c4_files = c4_utils.get_c4_files(path)
args = (c4_files)

name = 'c4_r_denoiser'
c4_helper(make_mlm_task, name, *args)

name = 'c4_s_denoiser'
c4_helper(make_plm_task, name, *args)

name = 'c4_x_denoiser'
c4_helper(make_mlm_task, name, *args, **{"noise_density": 0.5, "mean_noise_span_length": 32})

name = 'c4_causal_lm'
c4_helper(make_clm_task, name, *args)

seqio.MixtureRegistry.add(
    "c4_ul2",
    ["c4_r_denoiser", "c4_s_denoiser", "c4_x_denoiser"],
    default_rate=1
    )

seqio.MixtureRegistry.add(
    "c4_ul2_causal_0_50",
    [("c4_ul2", 0.50), ("c4_causal_lm", 0.50)],
    )