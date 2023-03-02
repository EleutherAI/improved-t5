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

from t5x.data.vocab import DEFAULT_OUTPUT_FEATURES

from t5x.data.p3 import p3_utils

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

# ==================================== P3 ======================================
# Adapted from T-Zero

# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture = {key: [] for key in p3_utils.t0_train}
t0_eval_mixture = {key: [] for key in p3_utils.t0_eval}
mixture_cap = {}
for dataset_name, subset_name in p3_utils.all_templates.keys:
    if (dataset_name, subset_name) not in p3_utils.all_datasets:
        p3_utils.all_templates.remove(dataset_name, subset_name)
        continue

    cap = p3_utils.get_cap(dataset_name, subset_name)
    dataset = p3_utils.all_templates.get_dataset(dataset_name, subset_name)

    for template_name in dataset.all_template_names:
        # Add train and normal eval tasks
        template = p3_utils.all_templates.get_dataset(dataset_name, subset_name)[template_name]
        task_name = "p3_"+p3_utils.get_task_name(dataset_name, subset_name, template_name)
        TaskRegistry.add(
            name=task_name,
            source=p3_utils.get_p3_source(dataset_name, subset_name, template_name),
            preprocessors=[
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
                seqio.CacheDatasetPlaceholder(required=False),
            ],
            postprocess_fn=p3_utils.maybe_get_class_id_postprocessor(template),
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=p3_utils.get_p3_metric(dataset_name, subset_name, template_name)
        )

        # # Add rank classification eval task
        # if template.answer_choices:
        #     rank_classification_preprocessor = functools.partial(
        #         t5.data.preprocessors.rank_classification,
        #         inputs_fn=lambda ex: tf.fill((len(ex["answer_choices"]),), ex["inputs"]),
        #         targets_fn=lambda ex: ex["answer_choices"],
        #         is_correct_fn=lambda ex: tf.equal(ex["answer_choices"], tf.strings.strip(ex["targets"])),
        #         weight_fn=lambda ex: 1.0,
        #     )
        #     fixed_choices = template.get_fixed_answer_choices_list()
        #     num_classes = len(fixed_choices) if fixed_choices else None
        #     seqio.TaskRegistry.add(
        #         task_name + "_score_eval",
        #         data_source,
        #         preprocessors=[rank_classification_preprocessor] + preprocessors,
        #         output_features=output_features,
        #         metric_fns=[functools.partial(t5.evaluation.metrics.rank_classification, num_classes=num_classes)],
        #         postprocess_fn=t5.data.postprocessors.rank_classification,
        #     )

        # Check that the dataset_subset_tuple is in t0_train
        for key, dataset_subset_tuples in p3_utils.t0_train.items():
            if (dataset_name, subset_name) in dataset_subset_tuples:
                t0_train_mixture[key].append(task_name)
                mixture_cap[task_name] = cap


MixtureRegistry.add(
    "t0_train",
    [task for task in t0_train_mixture["BASE"] \
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

MixtureRegistry.add(
    "t0+_train",
    [task for task in t0_train_mixture["BASE"] \
                    + t0_train_mixture["GPT_EVAL"] 
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: p3_utils.mixture_cap[t.name],
)

MixtureRegistry.add(
    "t0++_train",
    [task for task in t0_train_mixture["BASE"] \
                    + t0_train_mixture["GPT_EVAL"] \
                    + t0_train_mixture["SGLUE"] \
                        if task not in p3_utils.TASK_BLACKLIST],
    default_rate=lambda t: p3_utils.mixture_cap[t.name],
)