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
import functools
import t0.seqio_tasks as t0_tasks

from data.vocab import DEFAULT_OUTPUT_FEATURES

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== P3 ======================================
# Re-add Tasks to use with different vocab in DEFAULT_OUTPUT_FEATURES

all_task_names = list(TaskRegistry.names()).copy()

for task_name in all_task_names:

    original_task = seqio.get_mixture_or_task(task_name)
    TaskRegistry.remove(task_name)
    TaskRegistry.add(
        name=original_task.name,
        source=original_task.source,
        preprocessors=original_task.preprocessors,
        postprocess_fn=original_task.postprocess_fn,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=original_task.metric_fns
    )

MixtureRegistry.remove("t0_eval_score_eval")
MixtureRegistry.add(
    "t0_eval_score_eval",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval")
        and task.split("_score_eval")[0] in t0_tasks.tasks.t0_eval_mixture["BASE"]
        and task.split("_score_eval")[0] not in t0_tasks.tasks.TASK_BLACKLIST
        and "story_cloze" not in task
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)