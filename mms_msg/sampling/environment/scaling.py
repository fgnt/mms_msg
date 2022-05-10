import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from mms_msg.sampling.utils.rng import get_rng_example

logger = logging.getLogger('meeting_generator')


__all__ = [
    'sample_scaling_uniform',
    'UniformScalingSampler',
    'ConstantScalingSampler',
]


def _normalize_log_weights(weights):
    weights -= weights.mean()
    return weights.tolist()


def sample_scaling_uniform(example, *, max_weight: float = 5):
    """
    Gives each utterance a different log_weight, sampled from a uniform
    distribution between 0 and `max_weight` and mean-normalized.
    """
    rng = get_rng_example(example, 'log_weights')
    weights = rng.uniform(size=len(example['speaker_id']), high=max_weight)
    example['log_weights'] = _normalize_log_weights(weights)
    return example


@dataclass(frozen=True)
class UniformScalingSampler:
    max_weight: float = 5

    def __call__(self, example):
        return sample_scaling_uniform(example, max_weight=self.max_weight)


@dataclass(frozen=True)
class ConstantScalingSampler:
    weights: [float, List[float]] = 0.

    def __call__(self, example):
        weights = self.weights
        if not isinstance(weights, (list, tuple)):
            weights = [weights] * len(example['speaker_id'])
        weights = np.asarray(weights[:len(example['speaker_id'])])
        example['log_weights'] = _normalize_log_weights(weights)
        return example
