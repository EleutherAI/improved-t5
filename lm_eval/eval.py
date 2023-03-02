import re
import os

import ast
import json

import pandas as pd

from lm_eval import evaluator

from model import T5DecoderLM


batch_size = 64
device = "cuda"
num_fewshot = 4
task_names = ["hellaswag", "boolq", "mrpc"]
model_name = "hf/large_mixed_objective"

model_args="pretrained={}".format(model_name)
print("Building Model, {}".format(model_args))
model = T5DecoderLM.create_from_arg_string(
    model_args, {"batch_size": batch_size, "device": device}
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=task_names,
    num_fewshot=num_fewshot,
    batch_size=batch_size,
    device=device,
    no_cache=True,
)

print(results)

# results['config']['model'] = "model_name"
# results['config']['model_args'] = model_args

# dumped = json.dumps(results, indent=2)
# output_dict_dir = os.path.join(
#     output_dir,
#     "json",
#     model_size,
#     "term_frequency-{}-{}-{}shot.json".format(model_size, checkpoint, str(n).zfill(2))
# )
# with open(output_dict_dir, "w") as f:
#     f.write(dumped)

# for task in task_names:
#     results_dict = {
#         "model": model_name,
#         "checkpoint": checkpoint,
#         "task": task.EVAL_HARNESS_NAME,
#         "fewshot": n,
#         **results['results'][task.EVAL_HARNESS_NAME]
#         }

#     all_results_df = pd.concat(
#         [all_results_df, pd.Series(results_dict).to_frame().T],
#         ignore_index=True
#         )