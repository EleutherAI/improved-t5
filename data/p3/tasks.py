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
