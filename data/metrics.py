import seqio
import numpy as np

from typing import Sequence

def perplexity(targets: Sequence[str], scores: Sequence[int]):

    cross_entropy = -np.mean(scores)/len(targets)
    perplexity = np.exp(cross_entropy)

    return {
        "perplexity": seqio.metrics.Scalar(perplexity)
    }
