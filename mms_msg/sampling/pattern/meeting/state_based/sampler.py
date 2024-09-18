import numpy as np
import padertorch as pt

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

from mms_msg.sampling.utils.distribution_model import DistributionModel
from mms_msg.sampling.utils import collate_fn
from mms_msg.sampling.pattern.meeting.overlap_sampler import _get_valid_overlap_region


class SilenceSampler(ABC):
    """
    Abstract class to allow sampling of an integer value according to a certain distribution.
    Optionally the sampling can be restricted by a minimum and a maximum bound. This bound is guaranteed,
    when the sampler is called with (). Alternatively a value can be directly sampled with sample_silence,
    but then the bounds are not enforced. When the sampling fails a ValueError should be raised.

    Properties:
        hard_minimum_value: (optional) minimum value that should be sampled
        hard_maximum_value: (optional) maximum value that should be sampled
    """
    hard_minimum_value: int = 0
    hard_maximum_value: int = 1000000

    def __call__(self, rng: np.random.random = np.random.default_rng(), minimum_value: Optional[int] = None,
                 maximum_value: Optional[int] = None) -> int:
        """
        Samples an integer as silence value according to both, the class bounds and those given as parameters.

        Args:
            rng: (optional) The numpy rng that should be used, the rng should generate a number in the interval [0,1)
                When not set a new uniform rng is used.
            minimum_value: (optional) minimum_value that should be sampled,
                is overwritten by the hard limits of the class.
            maximum_value: (optional) minimum_value that should be sampled,
                is overwritten by the hard limits of the class.

        Returns: Integer that is guaranteed to be in the both given bounds, the class bounds and the parameter bounds.
        """
        if minimum_value is None:
            minimum_value = self.hard_minimum_value
        else:
            minimum_value = max(minimum_value, self.hard_minimum_value)

        if maximum_value is None:
            maximum_value = self.hard_maximum_value
        else:
            maximum_value = max(maximum_value, self.hard_maximum_value)

        if minimum_value >= maximum_value:
            raise ValueError('The maximum value must be greater than the minimum value. You have the change either'
                             ' the hard bounds of the class or the parameter bounds used when calling the sampler.')

        return self.sample_silence(rng, minimum_value, maximum_value)

    @abstractmethod
    def sample_silence(self, rng: np.random.random = np.random.default_rng(), minimum_value: Optional[int] = None,
                       maximum_value: Optional[int] = None) -> int:
        """
        Samples an integer according to the given bounds.

        Args:
            rng: (optional) The numpy rng that should be used, the rng should generate a number in the interval [0,1)
                When not set a new uniform rng is used.
            minimum_value: (optional) minimum_value that should be sampled.
            maximum_value: (optional) minimum_value that should be sampled.

        Returns: Integer that is guaranteed to be in the given bounds.
        """
        raise NotImplementedError()


@dataclass
class UniformSilenceSampler(SilenceSampler):
    """
    Generate uniform integer samples between a given min and max value.
    Optionally the sampling can be restricted by a minimum and a maximum bound. This bound is guaranteed,
    when the sampler is called with (). Alternatively a value can be directly sampled with sample_silence,
    but then the bounds are not enforced. When the sampling fails a ValueError is raised.

    Properties:
        hard_minimum_value: (optional) minimum value that should be sampled
        hard_maximum_value: (optional) maximum value that should be sampled
    """

    def sample_silence(self, rng: np.random.random = np.random.default_rng(), minimum_value: Optional[int] = None,
                       maximum_value: Optional[int] = None) -> int:
        return rng.integers(minimum_value, maximum_value)


class DistributionSilenceSampler(SilenceSampler):
    """
    Generates samples using a given distribution.
    Optionally the sampling can be restricted by a minimum and a maximum bound. This bound is guaranteed,
    when the sampler is called with (). Alternatively a value can be directly sampled with sample_silence,
    but then the bounds are not enforced. When the sampling fails a ValueError is raised.
    Properties:
        distribution: Distribution form which the silence values are sampled.
        minimum_value: Minimum value that can be sampled
        maximum_value: Maximum value that can be sampled
    """

    def __init__(self, distribution, minimum_value=None, maximum_value=None):
        """
        Initializes the sampler with a given distribution and optional maximum and minimum values.

        Args:
            distribution: Distribution form which the silence values are sampled.
            minimum_value: (optional) Hard minimum value that can be sampled, when not set this is influenced by the
                underlying distribution model.
            maximum_value: (optional) Hard maximum value that can be sampled, when not set this is influenced by the
                underlying distribution model.
        """
        if minimum_value is None:
            self.minimum_value = distribution.min_value
        else:
            self.minimum_value = minimum_value

        if maximum_value is None:
            self.maximum_value = distribution.max_value
        else:
            self.maximum_value = maximum_value

        self.distribution = distribution

    def sample_silence(self, rng: np.random.random = np.random.default_rng(), minimum_value: Optional[int] = None,
                       maximum_value: Optional[int] = None) -> int:
        return self.distribution.sample_value(rng, minimum_value=minimum_value, maximum_value=maximum_value)


@dataclass(frozen=False)
class OverlapSampler(pt.Configurable, ABC):
    """ Abstract class that allows to construct an Overlap sampler, which is used to sample overlap values for the
    generation of a meeting. It is guaranteed that the given overlap values are valid and only a maximum number
    of speakers is active simultaneously, when the sampler is called with ().
    The sampling process of the values must be implemented in a subclass.
    When the sampling fails a ValueError should be returned.

    Properties:
        max_concurrent_spk: Maximum number of concurrent active speakers.
        hard_minimum_overlap: Hard minimum value for the overlap
        hard_maximum_overlap: Hard maximum value for the overlap
    """
    max_concurrent_spk: int
    hard_minimum_overlap: int = 0
    hard_maximum_overlap: int = 1000000

    def __call__(self, examples: List[Dict], current_source: Dict[str, Any],
                 rng: np.random.random = np.random.default_rng(), use_vad: bool = False) -> int:
        """
        Determines the maximum allowed overlap and that samples an overlap value through the function _sample_overlap.

        Args:
            examples: List of all examples that are currently present in the meeting.
            current_source: Source for which the overlap should be determined.
            rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                When not set a uniform rng is used.
            use_vad: (optional) Is VAD data in the given datasets and should these data be used during
                the selection of samples and offset. Default Value: False

        Returns: Sampled overlap
        """

        maximum_overlap = _get_valid_overlap_region(collate_fn(examples), self.max_concurrent_spk, current_source, use_vad)
        examples = examples[:]

        if use_vad:
            examples.sort(key=lambda x: x['speaker_end']['aligned_source'])
            if len(examples) > 1:
                maximum_overlap = min(maximum_overlap,
                                      examples[-1]['speaker_end']['aligned_source']
                                      - examples[-2]['speaker_end']['aligned_source'])
            maximum_overlap = min(maximum_overlap, current_source['num_samples']['aligned_source'],
                                  examples[-1]['num_samples']['aligned_source'])
        else:
            maximum_overlap = min(maximum_overlap, current_source['num_samples']['observation'])

        maximum_overlap = min(maximum_overlap, self.hard_maximum_overlap)

        overlap = self._sample_overlap(self.hard_minimum_overlap, maximum_overlap, rng, examples, current_source)

        return overlap

    @abstractmethod
    def _sample_overlap(self, minimum_overlap: int, maximum_overlap: int,
                        rng: np.random.random = np.random.default_rng(), examples: List[Dict] = None,
                        current_source: Dict[str, Any] = None) -> int:
        """
        Internal function that samples overlap with respect to the maximum and minimum allowed overlap.
        Also, can take the previous examples and the current source as parameters
        when required for sampling the overlap.

        Args:
            minimum_overlap: Minimum for the overlap that is sampled
            maximum_overlap: Maximum for the overlap that is sampled
            rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                When not set a uniform rng is used.
            examples: (optional) List of all examples that are currently present in the meeting.
            current_source: (optional) Source for which the overlap should be determined.

        Returns: Sampled overlap
        """
        raise NotImplementedError


class DistributionOverlapSampler(OverlapSampler):
    """
    Class which is used to sample overlap values for the generation of a meeting using a DistributionModel.
    It is guaranteed that the given overlap values are valid and only a maximum number
    of speakers is active simultaneously, when the sampler is called with ().
    When the sampling fails a ValueError is returned.

    Properties:
        max_concurrent_spk: Maximum number of concurrent active speakers.
        distribution: DistributionModel from which the overlap should be sampled.
        hard_minimum_overlap: Hard minimum value for the overlap
        hard_maximum_overlap: Hard maximum value for the overlap
    """

    def __init__(self, max_concurrent_spk: int, distribution: DistributionModel, hard_minimum_overlap: int = 0,
                 hard_maximum_overlap: int = 1000000):
        """
        Args:
            max_concurrent_spk: Maximum number of concurrent active speakers.
            distribution: DistributionModel from which the overlap should be sampled.
            hard_minimum_overlap: Hard minimum value for the overlap
            hard_maximum_overlap: Hard maximum value for the overlap
        """

        self.max_concurrent_spk = max_concurrent_spk
        self.distribution = distribution
        self.hard_minimum_overlap = hard_minimum_overlap
        self.hard_maximum_overlap = hard_maximum_overlap

    def _sample_overlap(self, minimum_overlap: int, maximum_overlap: int,
                        rng: np.random.random = np.random.default_rng(), examples: List[Dict] = None,
                        current_source: Dict[str, Any] = None) -> int:
        """
        Internal function that samples overlap from the distribution with respect to the maximum
        and minimum allowed overlap. examples and current_source are not used

        Args:
            maximum_overlap: Maximum for the overlap that is sampled
            minimum_overlap: Minimum for the overlap that is sampled
            rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                When not set a uniform rng is used.
            examples: (Not used in the implementation of this function)
            current_source: (Not used in the implementation of this function)

        Returns: Sampled overlap according to the distribution
        """

        return self.distribution.sample_value(rng, minimum_value=minimum_overlap, maximum_value=maximum_overlap)


@dataclass
class BackchannelStartSampler:
    """
    Class that can be used for the sampling the starting distance of the backchannel source from the beginning of the
    foreground source.
    Example: Foreground offset: 2000, start_distance: 1500 => Backchannel offset: 3500

    Important: The current implementation does not guarantee, that the sampled start distance for
               the backchannel is valid, that must be ensured through the given parameters.

    Properties:
        minimum_start_distance: Hard minimum value for the start distance
        maximum_start_distance: Hard maximum value for the start distance
    """
    minimum_start_distance: int = 0
    maximum_start_distance: int = 16000000

    def __call__(self, minimum_possible_start: int, maximum_possible_start: int,
                 rng: np.random.random = np.random.default_rng()) -> int:
        """
        Samples the offset of a backchannel example, while making sure that the hard minimum and maximum distances
        are followed. Do not guarantee that the sampled offset is valid that has to be ensured
        through the input parameters. The offset is sampled uniformly in the possible range.

        Args:
            minimum_possible_start: Minimum possible offset for the backchannel source
            maximum_possible_start: Maximum possible offset for the backchannel source
            rng: The numpy rng that should be used, the rng should generate a number in the interval [0,1).
                When not set a uniform rng is used.

        Returns: Offset of the backchannel source
        """

        maximum_start = min(minimum_possible_start+self.maximum_start_distance, maximum_possible_start)
        minimum_start = min(minimum_possible_start+self.minimum_start_distance, self.maximum_start_distance)

        return rng.integers(minimum_start, maximum_start)
