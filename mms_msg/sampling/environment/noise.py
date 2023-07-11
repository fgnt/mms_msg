from dataclasses import dataclass

from typing import Tuple

from mms_msg.sampling.utils.rng import get_rng_example

__all__ = [
    'sample_uniform_snr',
    'UniformSNRSampler',
]

def sample_uniform_snr(example, *, min_snr: float = 20, max_snr: float = 30):
    example['snr'] = float(
        get_rng_example(example, 'snr').uniform(min_snr, max_snr, size=1)
    )
    return example


@dataclass(frozen=True)
class UniformSNRSampler:
    min_snr: float = 20
    max_snr: float = 30

    def __call__(self, example):
        return sample_uniform_snr(example, min_snr=self.min_snr, max_snr=self.max_snr)
