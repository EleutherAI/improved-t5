import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


model_dir_template = "/fsx/lintangsutawika/improved_t5/ckpts/920m_pile_ul2_causal_0_{}/"
metric_extension = "-metrics.jsonl"
model_checkpoints = [
    # "sglue_32k",
    # "sglue_64k",
    # "sglue_96k",
    # "sglue_128k",
    # "sglue_160k",
    # "sglue_192k",
    # "sglue_224k",
    "sglue_256k",
    ]

inference_result_path = "inference_eval/"

# metrics = ["accuracy", "em", "f1", "mean_3class_f1", "exact_match"]

sglue_score_df = pd.DataFrame()

flop_pretrain_rate = {
    'sglue_32k': 32 * 1000 * 2048 * 920 * 1000000,
    'sglue_64k': 64 * 1000 * 2048 * 920 * 1000000,
    'sglue_96k': 96 * 1000 * 2048 * 920 * 1000000,
    'sglue_128k': 128 * 1000 * 2048 * 920 * 1000000,
    'sglue_160k': 160 * 1000 * 2048 * 920 * 1000000,
    'sglue_192k': 192 * 1000 * 2048 * 920 * 1000000,
    'sglue_224k': 224 * 1000 * 2048 * 920 * 1000000,
    'sglue_256k': 256 * 1000 * 2048 * 920 * 1000000,
}

flop_rate = {
    10: 7596.666667,
    15: 7811,
    25: 8239.666667,
    50: 9311.333333,
    60: 9740,
    75: 10383
}

for rate in [10, 15, 25, 50, 60, 75]:

    model_dir = model_dir_template.format(rate)
    _rate = "0.{}".format(rate)

    for model in model_checkpoints:

        full_path = os.path.join(
            model_dir,
            model,
            inference_result_path
            )

        pretraining_step = int(model.split("_")[1][:-1])*1000

        try:
            for metric_file in os.listdir(full_path):
                if metric_extension in metric_file:
                    task_name = metric_file.split(metric_extension)[0]
                    with open(os.path.join(full_path, metric_file), "r") as file:
                        all_lines = file.readlines()

                        for line in all_lines:
                            _dict = json.loads(line)
                            step = _dict["step"]
                            _df = pd.DataFrame(
                                data={
                                        "model": model,
                                        "score": np.average([_dict[key] for key in _dict.keys() if key != "step"]),
                                            "step": step,
                                        "rate": _rate,
                                        "flops": 6*flop_pretrain_rate[model]*flop_rate[rate] + 6 * (step-pretraining_step) * 128 * 920 * 1000000 * flop_rate[rate]
                                    },
                                index=[0]
                                )

                            sglue_score_df = pd.concat([sglue_score_df, _df], ignore_index=True)

        except:
            pass

sglue_score_df = sglue_score_df.groupby(['model', 'step', 'rate', 'flops']).mean().reset_index()

# for _rate in [10, 15, 25, 50, 60, 75]:

#     rate = "0.{}".format(_rate)
#     for step in sglue_score_df[(sglue_score_df['rate'] == rate)]['step'].unique():
#         step_rows = sglue_score_df[
#                 (sglue_score_df['step'] == step) & \
#                     (sglue_score_df['rate'] == rate)
#             ]

#         max_score = step_rows['score'].max()
#         drop_index = list(step_rows['score'].index)

#         sglue_score_df = sglue_score_df.drop(drop_index)
#         _df = pd.DataFrame(
#             data={
#                 "model": "-",
#                 "score": max_score,
#                 "step": step,
#                 "rate": rate,
#                 "flops": flop_pretrain_rate[] + np.int64(6 * step * 128 * 920*10**6 * flop_rate[_rate])
#                 },
#             index=[0]
#             )
#         sglue_score_df = pd.concat([sglue_score_df, _df], ignore_index=True)

        
            

# sglue_score_df['score'] = sglue_score_df[metrics].mean(axis=1)
# graph_df = sglue_score_df.groupby(['model', 'pretraining_step', 'flops'])['score'].mean().reset_index()

# graph_df = sglue_score_df.groupby(['model', 'step', '])['score'].mean().reset_index()

sns.lineplot(
    data=sglue_score_df,
    x="step",
    y="score",
    hue="rate",
    # hue="model",
    # hue_order=model_checkpoints,
    # markers=True, dashes=False
)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('super_glue_performance_flop.png', dpi=200)
plt.clf()



sns.scatterplot(
    data=sglue_score_df, #[sglue_score_df['rate'] == '0.15'],
    x="flops",
    y="score",
    hue="rate",
    # hue="model",
    # hue_order=model_checkpoints,
    # markers=True, dashes=False
)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('super_glue_performance_flop.png', dpi=200)
plt.clf()