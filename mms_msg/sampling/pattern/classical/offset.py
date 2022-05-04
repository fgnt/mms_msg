from dataclasses import dataclass

from mms_msg import keys
from mms_msg.sampling.utils.rng import get_rng_example
from mms_msg.sampling.utils.utils import update_num_samples


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


def sample_partial_overlap(example, *, minimum_overlap, maximum_overlap):
    """
    >>> from IPython.lib.pretty import pprint
    >>> ex = {
    ...     'dataset': 'dataset',  # Needed for rng
    ...     'example_id': 'example_id',  # Needed for rng
    ...     'num_samples': {'original_source': [10_000, 15_000]}
    ... }
    >>> ex = sample_partial_overlap(ex, minimum_overlap=0, maximum_overlap=1)
    >>> del ex['dataset'], ex['example_id']
    >>> pprint(ex)
    {'num_samples': {'original_source': [10000, 15000],
      'observation': 20390.74687131983},
     'offset': {'original_source': [0, 5390.746871319828]}}
    """
    rng = get_rng_example(example, 'offset')
    overlap = rng.uniform(minimum_overlap, maximum_overlap)
    num_samples = example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE]
    assert len(num_samples) == 2, (len(num_samples), num_samples)
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
    minimum_overlap: float
    maximum_overlap: float

    def __call__(self, example):
        return sample_partial_overlap(
            example,
            minimum_overlap=self.minimum_overlap,
            maximum_overlap=self.maximum_overlap
        )
