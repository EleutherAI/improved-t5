#
# Adapted from https://github.com/bigscience-workshop/t-zero/blob/master/t0/seqio_tasks/tasks.py


"""
This file defines 8 mixtures that we used in the T-Zero paper:
- t0_train: T0 training mixture
- t0+_train: T0+ training mixture
- t0++_train: T0++ training mixture
- t0_eval_score_eval: T0 main evaluation mixture (Figure 4 for instance)
- t0_train_score_eval: Evaluation mixture for checkpoint selection on T0 (validation splits of the training sets)
- t0_train_one_og_prompt: T0 (p=1) training mixture for  - one original-task prompt per dataset. Figure 6
- t0_train_all_og_prompts: T0 (p=5.7) training mixture for - all original-task prompts for all datasets. Figure 6
- bias_fairness_eval_score_eval: Bias & fairness evaluation mixture. Appendix B3
"""


import re
import seqio
import datasets
import functools
import tensorflow as tf

import promptsource.utils
from promptsource import templates

import t5
from t5.evaluation import metrics as mt
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric


MAX_EXAMPLES_PER_DATASET = 500_000

GET_METRICS = {
    "BLEU": mt.bleu,
    "ROUGE": mt.rouge,
    "Span Squad": mt.span_squad,
    "Squad": mt.squad,
    "Trivia QA": mt.trivia_qa,
    "Accuracy": mt.accuracy,
    "Sequence Accuracy": mt.sequence_accuracy,
    "Pearson Correlation": mt.pearson_corrcoef,
    "Spearman Correlation": mt.spearman_corrcoef,
    "MultiRC": mt.multirc_f1_over_all_answers,
    "AUC": mt.auc,
    "COQA F1": mt.coqa_f1,
    "Edit Distance": mt.edit_distance,
    # "Mean Reciprocal Rank": mt.accuracy,  # NOTE not in T5?
    "Other": mt.accuracy,
    # Missing support for mean_multiclass_f1 etc. which need a num_classes parameter
}

t0_eval = {
    'BASE': [
        ('super_glue', 'wsc.fixed'),
        ('winogrande', 'winogrande_xl'),
        ('super_glue', 'cb'),
        ('super_glue', 'rte'),
        ('anli', None),
        ('super_glue', 'copa'),
        ('story_cloze', '2016'),
        ('hellaswag', None),
        ('super_glue', 'wic')
    ],
    'BIAS_FAIRNESS': [
        ('crows_pairs', None),
        ('jigsaw_toxicity_pred', None),
        ('super_glue', 'axg'),
        ('wino_bias', 'type1_anti'),
        ('wino_bias', 'type2_anti'),
        ('wino_bias', 'type1_pro'),
        ('wino_bias', 'type2_pro')
    ]
}

t0_train = {
    'BASE': [
        ('glue', 'mrpc'),
        ('glue', 'qqp'),
        ('paws', 'labeled_final'),
        ('kilt_tasks', 'hotpotqa'),
        ('wiki_qa', None),
        ('adversarial_qa', 'dbidaf'),
        ('adversarial_qa', 'dbert'),
        ('adversarial_qa', 'droberta'),
        ('duorc', 'SelfRC'),
        ('duorc', 'ParaphraseRC'),
        ('ropes', None),
        ('quoref', None),
        ('cos_e', 'v1.11'),
        ('cosmos_qa', None),
        ('dream', None),
        ('qasc', None),
        ('quail', None),
        ('quarel', None),
        ('quartz', None),
        ('sciq', None),
        ('social_i_qa', None),
        ('wiki_hop', 'original'),
        ('wiqa', None),
        ('amazon_polarity', None),
        ('app_reviews', None),
        ('imdb', None),
        ('rotten_tomatoes', None),
        ('yelp_review_full', None),
        ('common_gen', None),
        ('wiki_bio', None),
        ('cnn_dailymail', '3.0.0'),
        ('gigaword', None),
        ('multi_news', None),
        ('samsum', None),
        ('xsum', None),
        ('ag_news', None),
        ('dbpedia_14', None),
        ('trec', None)
    ],
 'GPT_EVAL': [
        ('ai2_arc', 'ARC-Challenge'),
        ('ai2_arc', 'ARC-Easy'),
        ('trivia_qa', 'unfiltered'),
        ('web_questions', None),
        ('squad_v2', None),
        ('openbookqa', 'main'),
        ('race', 'high'),
        ('race', 'middle'),
        ('piqa', None),
        ('hellaswag', None)
    ],
 'SGLUE': [
        ('super_glue', 'wsc.fixed'),
        ('super_glue', 'record'),
        ('super_glue', 'boolq'),
        ('super_glue', 'copa'),
        ('super_glue', 'multirc'),
        ('super_glue', 'wic')
    ]
}

mixture_cap = {
    ("super_glue", "wsc.fixed"): 554,
    ("winogrande", "winogrande_xl"): 40398,
    ("super_glue", "cb"): 250,
    ("super_glue", "rte"): 2490,
    ("anli", None): 162865,
    ("glue", "mrpc"): 3668,
    ("glue", "qqp"): 363846,
    ("paws", "labeled_final"): 49401,
    ("ai2_arc", "ARC-Challenge"): 1119,
    ("ai2_arc", "ARC-Easy"): 2251,
    ("kilt_tasks", "hotpotqa"): 88869,
    ("trivia_qa", "unfiltered"): 87622,
    ("web_questions", None): 3778,
    ("wiki_qa", None): 20360,
    ("adversarial_qa", "dbidaf"): 10000,
    ("adversarial_qa", "dbert"): 10000,
    ("adversarial_qa", "droberta"): 10000,
    ("duorc", "SelfRC"): 60721,
    ("duorc", "ParaphraseRC"): 69524,
    ("ropes", None): 10924,
    ("squad_v2", None): 130319,
    ("super_glue", "record"): 100730,
    ("quoref", None): 19399,
    ("cos_e", "v1.11"): 9741,
    ("cosmos_qa", None): 25262,
    ("dream", None): 6116,
    ("openbookqa", "main"): 4957,
    ("qasc", None): 8134,
    ("quail", None): 10246,
    ("quarel", None): 1941,
    ("quartz", None): 2696,
    ("race", "high"): 62445,
    ("race", "middle"): 25421,
    ("sciq", None): 11679,
    ("social_i_qa", None): 33410,
    ("super_glue", "boolq"): 9427,
    ("super_glue", "copa"): 400,
    ("super_glue", "multirc"): 27243,
    ("wiki_hop", "original"): 43738,
    ("wiqa", None): 29808,
    ("piqa", None): 16113,
    ("amazon_polarity", None): 3600000,
    ("app_reviews", None): 288065,
    ("imdb", None): 25000,
    ("rotten_tomatoes", None): 8530,
    ("yelp_review_full", None): 650000,
    ("hellaswag", None): 39905,
    ("common_gen", None): 67389,
    ("wiki_bio", None): 582659,
    ("cnn_dailymail", "3.0.0"): 287113,
    ("gigaword", None): 3803957,
    ("multi_news", None): 44972,
    ("samsum", None): 14732,
    ("xsum", None): 204045,
    ("ag_news", None): 120000,
    ("dbpedia_14", None): 560000,
    ("trec", None): 5452,
    ("super_glue", "wic"): 5428,
}

TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # Tasks with broken cached files
    "gigaword_summarize_",
]

# Tasks that failed caching (won't try to fix them for now) - remove when we are done
D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST = [
    "amazon_polarity_Is_this_product_review_positive_score_eval",
    "amazon_polarity_Is_this_review_negative_score_eval",
    "amazon_polarity_Is_this_review_score_eval",
    "amazon_polarity_User_recommend_this_product_score_eval",
    "amazon_polarity_convey_negative_or_positive_sentiment_score_eval",
    "amazon_polarity_flattering_or_not_score_eval",
    "amazon_polarity_negative_or_positive_tone_score_eval",
    "amazon_polarity_user_satisfied_score_eval",
    "amazon_polarity_would_you_buy_score_eval",
    "dbpedia_14_given_a_choice_of_categories__score_eval",
    "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to_score_eval",
    "dbpedia_14_pick_one_category_for_the_following_text_score_eval",
    "wiki_hop_original_choose_best_object_affirmative_1_score_eval",
    "wiki_hop_original_choose_best_object_affirmative_2_score_eval",
    "wiki_hop_original_choose_best_object_affirmative_3_score_eval",
    "wiki_hop_original_choose_best_object_interrogative_1_score_eval",
    "wiki_hop_original_choose_best_object_interrogative_2_score_eval",
]

all_datasets = sum(t0_train.values(), []) + sum(t0_eval.values(), [])
all_templates = templates.TemplateCollection()
all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions

def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(shape=() if not length else (None if length == -1 else length,), dtype=tf.int64)
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=getattr(tf.dtypes, feature.dtype)
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def hf_dataset_to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        dataset.__iter__, output_signature={k: feature_to_spec(v) for k, v in dataset.features.items()}
    )


def apply_template(dataset, template):
    def map_fn(ex):
        ex = promptsource.utils.removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)


def strip_whitespace(output_or_target, example=None, is_target=False):
    """Cached tasks from promptsource all have a leading space on the ground-truth targets."""
    return output_or_target.strip()


def maybe_get_class_id_postprocessor(template):
    if template.get_fixed_answer_choices_list():

        def postprocess_fn(output_or_target, example=None, is_target=False):
            output_or_target = strip_whitespace(output_or_target)
            return t5.data.postprocessors.string_label_to_class_id(
                output_or_target, label_classes=template.get_fixed_answer_choices_list()
            )

        return postprocess_fn

    else:
        return strip_whitespace


def get_tf_dataset(split, shuffle_files, seed, dataset_name=None, subset_name=None, template=None, split_mapping=None):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = datasets.load_dataset(dataset_name, subset_name)
    dataset = dataset[split_mapping[split]]
    dataset = apply_template(dataset, template)
    return hf_dataset_to_tf_dataset(dataset)


def get_cap(dataset_name, subset_name):

    dataset = all_templates.get_dataset(dataset_name, subset_name)
    num_templates = len(dataset.all_template_names)

    if (dataset_name, subset_name) in mixture_cap:
        train_size = mixture_cap[(dataset_name, subset_name)]
    else:
        train_size = 0

    if train_size > MAX_EXAMPLES_PER_DATASET:
        cap = MAX_EXAMPLES_PER_DATASET // num_templates
    else:
        cap = train_size
    
    return cap

def get_p3_metric(dataset_name, subset_name, template_name):

    if dataset_name == "glue":
        metrics = get_glue_metric(subset_name)
    elif dataset_name == "super_glue":
        if subset_name in ("wsc.fixed", "multirc"):
            # TODO: WSC and MultiRC need special pre/postprocesing
            metrics = [mt.accuracy]
        else:
            metrics = get_super_glue_metric(subset_name)
    else:
        # TODO what if metric is null?
        template = all_templates.get_dataset(dataset_name, subset_name)[template_name]
        metrics = [GET_METRICS[m] for m in template.metadata.metrics]

    return metrics

def get_p3_source(dataset_name, subset_name, template_name):

    dataset_splits = get_dataset_splits(dataset_name, subset_name)
    split_mapping = None or {k: k for k in dataset_splits.keys()}

    template = all_templates.get_dataset(dataset_name, subset_name)[template_name]

    dataset_fn = functools.partial(
        get_tf_dataset,
        seed=None,
        dataset_name=dataset_name,
        subset_name=subset_name,
        template=template,
        split_mapping=split_mapping,
    )
    data_source = seqio.FunctionDataSource(
        dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: dataset_splits[split_mapping[s]].num_examples for s in split_mapping.keys()},
    )

    return data_source


