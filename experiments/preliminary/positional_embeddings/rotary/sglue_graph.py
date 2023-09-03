import os
import json

import pandas as pd
# import seaborn as sns


model_dir = "/fsx/lintangsutawika/improved_t5/ckpts/rotary/"
metric_extension = "-metrics.jsonl"
model_checkpoints = [
    "benchmark_pile_mlm",
    "rotary_pile_mlm",
    "rotary_relpos_pile_mlm",
    ]

inference_result_path = "inference_eval/"

metrics = ["accuracy", "em", "f1", "mean_3class_f1", "exact_match"]

sglue_score_df = pd.DataFrame()

for model in model_checkpoints:

    full_path = os.path.join(
        model_dir,
        model,
        "finetune_sglue",
        inference_result_path
        )

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
                        **_dict,
                        },
                    index=[0]
                    )

                sglue_score_df = pd.concat([sglue_score_df, _df], ignore_index=True)

sglue_score_df['score'] = sglue_score_df[metrics].mean(axis=1)
graph_df = sglue_score_df.groupby(['model', 'task_name'])['score'].mean().reset_index()

# sns.lineplot(
#     data=graph_df,
#     # x="pretraining_step",
#     x="flops",
#     y="score",
#     hue="model",
#     hue_order=["1_6b", "920m", "470m", "200m", "110m", "60m", "25m"],
#     markers=True, dashes=False
# )
# plt.xscale('log')
# # plt.yscale('log')
# plt.savefig('super_glue_performance_flop.png', dpi=200)
# plt.clf()

# In [38]: experiment_id = "qaavJm4oS6y7znwDOmRA7w"

# In [39]: experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
#     ...: df = experiment.get_scalars()
#     ...: df
# In [40]: _df = df[
#     ...:     (df['run'].str.contains('/training_eval')) & \
#     ...:     (df['tag'] == 'loss_per_nonpadding_target_token') & \
#     ...:     # (~df['run'].str.contains('eval')) & \
#     ...:     (~df['run'].str.contains('vanilla'))
#     ...:     ]

# In [41]: _df.to_csv("pretraining_loss.csv", index=False)


# df = pd.read_csv("pretraining_loss.csv")

# df['num_seq'] = df['step'] * 2048
# df['param'] = 0
# df['param'][df['run'].str.contains('25m')] = 25*10**6
# df['param'][df['run'].str.contains('60m')] = 60*10**6
# df['param'][df['run'].str.contains('110m')] = 110*10**6
# df['param'][df['run'].str.contains('200m')] = 200*10**6
# df['param'][df['run'].str.contains('470m')] = 470*10**6
# df['param'][df['run'].str.contains('920m')] = 920*10**6
# df['param'][df['run'].str.contains('1_6b')] = 1600*10**6

# df['model'] = ""
# df['model'][df['run'].str.contains('25m')] = "25m"
# df['model'][df['run'].str.contains('60m')] = "60m"
# df['model'][df['run'].str.contains('110m')] = "110m"
# df['model'][df['run'].str.contains('200m')] = "200m"
# df['model'][df['run'].str.contains('470m')] = "470m"
# df['model'][df['run'].str.contains('920m')] = "920m"
# df['model'][df['run'].str.contains('1_6b')] = "1_6b"

# # df['flops'] = (3 * df['num_seq'] * 512 * df['param']).astype(np.longdouble)
# df['flops'] = 10**(np.log10(3 * 512) + np.log10(df['num_seq']) + np.log10(df['param']))

# df = df.drop(index=df[df['run'].str.contains('200m') & (df['step'] < 1000)].index)

# df['validation loss'] = df['value']

# sns.lineplot(
#     data=df,
#     x="flops",
#     y="validation loss",
#     hue="model",
#     hue_order=["1_6b", "920m", "470m", "200m", "110m", "60m", "25m"],
#     markers=True, dashes=False
# )
# # plt.loglog(x, y, basex=10, basey=2)
# plt.yscale('log')
# plt.xscale('log', base=10)
# plt.savefig('pretraining_performance_flop.png', dpi=200)
# plt.clf()


# _df = df[
#     (df['run'].str.contains('/training_eval')) & \
#     (df['tag'] == 'loss_per_nonpadding_target_token') & \
#     # (~df['run'].str.contains('eval')) & \
#     (~df['run'].str.contains('vanilla'))
#     ]
# _df.to_csv("pretraining_loss.csv", index=False)