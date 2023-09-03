import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


model_dir = "/fsx/aran/jax/ckpts/scaling"
metric_extension = "-metrics.jsonl"
model_checkpoints = {
    # "25m": "vanilla_64k_finetune/", # 192000
    "60m": "vanilla_64k_finetune/", # 192000
    # "110m": "vanilla_80k_finetune/", # 192000
    "200m": "vanilla_128k_finetune/", # 256000
    "470m": "vanilla_256k_finetune/", # 384000
    # "920m",
    # "1_6b",
}

inference_result_path = "inference_eval/"

metrics = ["accuracy", "em", "f1", "mean_3class_f1", "exact_match"]

sglue_score_df = pd.DataFrame()

for model, checkpoint_path in model_checkpoints.items():

    full_path = os.path.join(
        model_dir,
        model,
        checkpoint_path,
        inference_result_path
    )

    for metric_file in os.listdir(full_path):
        if metric_extension in metric_file:
            task_name = metric_file.split(metric_extension)[0]
            with open(os.path.join(full_path, metric_file), "r") as file:
                all_lines = file.readlines()

                _dict = {}
                for idx, line in enumerate(all_lines):
                    for key, value in json.loads(line).items():
                        if idx == 0:
                            _dict[key] = [value]
                        else:
                            _dict[key].append(value)

                _df = pd.DataFrame(
                    data={
                            "model": [model]*len(_dict['step']),
                        "task_name": [task_name]*len(_dict['step']),
                        **_dict,
                        },
                    )

                sglue_score_df = pd.concat([sglue_score_df, _df], ignore_index=True)

sglue_score_df['score'] = sglue_score_df[metrics].mean(axis=1)
graph_df = sglue_score_df.groupby(['model', 'step'])['score'].mean().reset_index()

sns.lineplot(
    data=graph_df,
    x="step", y="score", hue="model",
    markers=True, dashes=False
)

plt.savefig('super_glue_performance.png', dpi=200)
plt.clf()
