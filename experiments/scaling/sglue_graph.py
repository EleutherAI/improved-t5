import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


model_dir = "/fsx/aran/jax/ckpts/scaling"
metric_extension = "-metrics.jsonl"
model_checkpoints = [
    "25m",
    "60m",
    "110m",
    "200m",
    "470m",
    "920m",
    "1_6b",
    ]

model_param = [
    25*10**6,
    60*10**6,
    100*10**6,
    200*10**6,
    470*10**6,
    920*10**6,
    1600*10**6,
]

inference_result_path = "inference_eval/"

metrics = ["accuracy", "em", "f1", "mean_3class_f1", "exact_match"]

sglue_score_df = pd.DataFrame()

for model, model_size in zip(model_checkpoints, model_param):

    model_path = os.path.join(
        model_dir,
        model
        )

    for checkpoint_path in sorted([_dir for _dir in os.listdir(model_path) if "vanilla_" in _dir]):
        full_path = os.path.join(
            model_path,
            checkpoint_path,
            inference_result_path
        )

        pretraining_step = int(checkpoint_path.split("_")[1][:-1])*1000

        num_tokens = pretraining_step * 2048 * 512

        for metric_file in os.listdir(full_path):
            if metric_extension in metric_file:
                task_name = metric_file.split(metric_extension)[0]
                with open(os.path.join(full_path, metric_file), "r") as file:
                    all_lines = file.readlines()
                    _dict = json.loads(all_lines[-1])

                    _df = pd.DataFrame(
                        data={
                                       "model": model,
                                   "task_name": task_name,
                            "pretraining_step": pretraining_step,
                                       "flops": int(3.5 * model_size * num_tokens),
                            **_dict,
                            },
                        index=[0]
                        )

                    sglue_score_df = pd.concat([sglue_score_df, _df], ignore_index=True)

sglue_score_df['score'] = sglue_score_df[metrics].mean(axis=1)
graph_df = sglue_score_df.groupby(['model', 'pretraining_step', 'flops'])['score'].mean().reset_index()

sns.lineplot(
    data=graph_df,
    # x="pretraining_step",
    x="flops",
    y="score",
    hue="model",
    hue_order=["1_6b", "920m", "470m", "200m", "110m", "60m", "25m"],
    markers=True, dashes=False
)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('super_glue_performance_flop.png', dpi=200)
plt.clf()
