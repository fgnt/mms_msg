from dataclasses import dataclass

import numpy as np

from padercontrib.database import keys
from .utils import get_rng_example, update_num_samples


def _assign_offset(example, offset):
    assert keys.OFFSET not in example
    example[keys.OFFSET] = {keys.ORIGINAL_SOURCE: offset}
    update_num_samples(example)
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
    _assign_offset(example, offset)
    return example


def sample_offsets_constant(example, *, offsets):
    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets] * len(example['speaker_id'])
    offset = list(offsets[:len(example['speaker_id'])])
    _assign_offset(example, offset)
    return example


def sample_partial_overlap(example, *, min_overlap, max_overlap):
    rng = get_rng_example(example, 'offset')
    overlap = rng.uniform(min_overlap, max_overlap)
    num_samples = example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE]
    assert len(num_samples) == 2
    overlap_samples = sum(num_samples)*overlap / (1 + overlap)
    offset = [0, max(num_samples[0] - overlap_samples, 0)]
    _assign_offset(example, offset)
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


@dataclass(frozen=True)
class PartialOverlapOffsetSampler:
    min_overlap: float
    max_overlap: float

    def __call__(self, example):
        return sample_partial_overlap(
            example,
            min_overlap=self.min_overlap,
            max_overlap=self.max_overlap
        )
