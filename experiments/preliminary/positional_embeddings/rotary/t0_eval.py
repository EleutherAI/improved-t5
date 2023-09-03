import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_dataset_name(prompt_name):
    dataset_list = [
        "super_glue_rte",
        "super_glue_cb",
        "anli",
        "super_glue_wsc",
        "winogrande",
        "super_glue_copa",
        "hellaswag",
        "super_glue_wic",
    ]

    for dataset in dataset_list:
        if dataset in prompt_name:
            break
    
    if dataset == "anli":
        for r_num in ["r1", "r2", "r3"]:
            if r_num in prompt_name:
                return "{}_{}".format(dataset, r_num)
    else:
        return dataset

model_dir_template = "/fsx/lintangsutawika/improved_t5/ckpts/METRO/"
metric_extension = "-metrics.jsonl"
model_checkpoints = [
    # "alibi_relpos_pile_mlm",
    # "benchmark_pile_mlm",
    # "metro_inspired_pile_mlm",
    "metro_inspired_learning_only_pile",
    "metro_no_alibi_pile"
    ]

inference_result_path = "finetune_t0_eval/inference_eval/"

# metrics = ["accuracy", "em", "f1", "mean_3class_f1", "exact_match"]

t0_eval_score_df = pd.DataFrame()

for model in model_checkpoints:

    full_path = os.path.join(
        model_dir_template,
        model,
        inference_result_path
        )

    for metric_file in os.listdir(full_path):
        if metric_extension in metric_file:
            task_name = metric_file.split(metric_extension)[0]
            with open(os.path.join(full_path, metric_file), "r") as file:
                all_lines = file.readlines()

                for line in all_lines:
                    _dict = json.loads(line)
                    step = _dict["step"]
                    score = _dict["accuracy"]
                    _df = pd.DataFrame(
                        data={
                                "run": model,
                                "score": score,
                                "dataset": get_dataset_name(task_name),
                                "task_name": task_name,
                            },
                        index=[0]
                        )

                    t0_eval_score_df = pd.concat([t0_eval_score_df, _df], ignore_index=True)

t0_eval_score_df = t0_eval_score_df.groupby(['run', 'dataset']).mean(['score']).reset_index()

