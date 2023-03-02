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

from data.pile import pile_utils
from data.utils import make_mlm_task, make_clm_task, make_plm_task

# ==================================== The Pile ======================================

PILE_SIZES = [(None, None), (20, '2b'), (5, '500m'), (1, '100m')]

path="/fsx/pile_raw_deduped/dedupe0.87/train/"

def pile_helper(task, name, pile_files, pile_size_name, **kwargs):
    if pile_size_name:
        name += f"_{pile_size_name}"
    task(name, pile_files, jsonl=True, **kwargs)

for num_files, pile_size_name in PILE_SIZES:
    if num_files is None:
        pile_files = pile_utils.get_pile_files(path=path)
    else:
        pile_files = pile_utils.get_minipile_files(path, num_files)
    args = (pile_files, pile_size_name)

    if pile_size_name is None:
        pile_size_name = ""
    else:
        pile_size_name = "_{}".format(pile_size_name)
    
    name = 'pile_r_denoiser{}'.format(pile_size_name)
    pile_helper(make_mlm_task, name, *args)

    name = 'pile_s_denoiser{}'.format(pile_size_name)
    pile_helper(make_plm_task, name, *args)

    name = 'pile_x_denoiser{}'.format(pile_size_name)
    pile_helper(make_mlm_task, name, *args, **{"noise_density": 0.5, "mean_noise_span_length": 32})

    name = 'pile_causal_lm{}'.format(pile_size_name)
    pile_helper(make_clm_task, name, *args)


seqio.MixtureRegistry.add(
    "pile_ul2",
    ["pile_r_denoiser", "pile_s_denoiser", "pile_x_denoiser"],
    default_rate=1
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_10",
    [("pile_ul2", 0.90), ("pile_causal_lm", 0.10)],
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_15",
    [("pile_ul2", 0.85), ("pile_causal_lm", 0.15)],
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_25",
    [("pile_ul2", 0.75), ("pile_causal_lm", 0.25)],
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_50",
    [("pile_ul2", 0.50), ("pile_causal_lm", 0.50)],
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_60",
    [("pile_ul2", 0.40), ("pile_causal_lm", 0.60)],
    )

seqio.MixtureRegistry.add(
    "pile_ul2_causal_0_75",
    [("pile_ul2", 0.25), ("pile_causal_lm", 0.75)],
    )
