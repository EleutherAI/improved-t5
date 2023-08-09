import collections
import functools

from t5.data.postprocessors import string_label_to_class_id, record


def get_natural_postprocess_fn(builder_config):

    benchmark_name = builder_config.name
    if benchmark_name == "record":
        return record
    elif benchmark_name == "multirc":    
        return multirc
    else:
        label_names = ["No", "Yes"]
        if benchmark_name == "cb":
            label_names = ["No", "Yes", "Maybe"]
        elif benchmark_name == "copa":
            label_names = builder_config.label_classes
        return functools.partial(
            string_label_to_class_id,
            label_classes=label_names,
        )

def multirc(string_label, example=None, is_target=False):
    """Returns dict containing the class with the question index for grouping."""
    res = {
        "value":
            string_label_to_class_id(
                string_label, example=example, label_classes=("No", "Yes"))
    }
    # Add the group, if present, since the model outputs will not have it.
    if is_target:
        res["group"] = example["idx/question"]
    return res
