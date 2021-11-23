from dataclasses import dataclass
import numpy as np
from ..utils import collate_fn


def _get_valid_overlap_region(examples, max_concurrent_spk, current_source):
    """
    Compute maximum overlap that guarantees that no more than  max_concurrent_spk are active at the same time.
    Note: This function underestimates the maximal overlap to ensure regions sampled as silence or
    repetitions of the same speaker will have no overlapping speech added later in the sampling   .
    Args:
        examples:
        max_concurrent_spk:

    Returns:

    """
    speaker_end = examples['speaker_end']
    speaker_id = examples['speaker_id']

    speaker_id = speaker_id + [None] * (max_concurrent_spk - len(speaker_end))
    speaker_end = speaker_end + [0] * (max_concurrent_spk - len(speaker_end))

    # Only keep end points relevant for the current sampling
    speaker_idx = np.argsort(speaker_end)[-max_concurrent_spk:]

    speaker_end = np.array(speaker_end)[speaker_idx]
    speaker_id = np.array(speaker_id)[speaker_idx]

    if current_source['speaker_id'] in speaker_id:
        spk_pos = list(speaker_id)[::-1].index(current_source['speaker_id'])
        max_concurrent_spk = spk_pos + 1  # remove index shift introduced by flipping the list
    max_overlap = speaker_end[-1] - speaker_end[-max_concurrent_spk]
    return max_overlap


@dataclass(frozen=True)
class OverlapSampler:
    max_concurrent_spk: int

    def __call__(self, examples, current_source, rng):
        examples = collate_fn(examples)
        maximum_overlap = _get_valid_overlap_region(examples, self.max_concurrent_spk, current_source)

        offset = self.sample_offset(examples, maximum_overlap, rng)

        return offset

    def sample_offset(self, examples, maximum_overlap, rng):
        return NotImplementedError


@dataclass(frozen=True)
class UniformOverlapSampler(OverlapSampler):
    p_silence: float
    maximum_silence: int
    maximum_overlap: int

    def sample_offset(self, examples, maximum_overlap, rng):
        def sample_shift():
            if rng.uniform(0, 1) <= self.p_silence:
                shift = self._sample_silence(rng)
            else:
                shift = self._sample_overlap(rng)
                shift = -1 * shift
            return shift

        speaker_end = sorted(examples['speaker_end'])
        shift = sample_shift()

        while shift < -1 * maximum_overlap:
            shift = sample_shift()

        offset = sorted(speaker_end)[-1] + shift
        return offset

    def _sample_silence(self, rng):
        silence = rng.integers(0, self.maximum_silence)
        return silence

    def _sample_overlap(self, rng):
        overlap = rng.integers(0, self.maximum_overlap)
        return overlap
