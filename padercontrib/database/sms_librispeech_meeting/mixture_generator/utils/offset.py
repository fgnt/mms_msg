from dataclasses import dataclass

import numpy as np
from .utils import get_rng_example


def _update_num_samples(example):
    example['num_samples']['observation'] = np.max(np.asarray(
        example['num_samples']['original_source']
    ) + np.asarray(example['offset']))
    return example


def sample_offsets_sms_wsj(example):
    offset = []
    rng = get_rng_example(example, 'offset')
    num_samples = example['num_samples']['original_source']
    total_length = max(num_samples)
    for ns in num_samples:
        excess_samples = total_length - ns
        assert excess_samples >= 0, excess_samples
        offset.append(rng.integers(0, excess_samples + 1))
    example['offset'] = offset
    _update_num_samples(example)
    return example


def sample_offsets_constant(example, *, offsets):
    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets] * len(example['speaker_id'])
    example['offset'] = list(offsets[:len(example['speaker_id'])])
    _update_num_samples(example)
    return example


@dataclass(frozen=True)
class ConstantOffsetSampler:
    """Samples constant offsets, e.g., for WSJ0-2mix-like data
    >>> ConstantOffsetSampler(0)({'speaker_id': 'abc'})
    """
    offsets: [int, list, tuple] = 0

    def __call__(self, example):
        return sample_offsets_constant(example, offsets=self.offsets)


@dataclass(frozen=True)
class SMSWSJOffsetSampler:
    def __call__(self, example):
        return sample_offsets_sms_wsj(example)
