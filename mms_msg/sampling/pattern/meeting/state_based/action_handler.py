from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from typing import Optional, Set, Dict, Union, Any, List, Tuple
from lazy_dataset import Dataset

from mms_msg.sampling.utils import sequence_sampling
from mms_msg.sampling.utils.rng import get_rng

from mms_msg.sampling.pattern.meeting.state_based.sampler import SilenceSampler, BackchannelStartSampler, OverlapSampler

logger = logging.getLogger('meeting_generation')


class ActionHandler(ABC):
    """
    Class for processing events during the generation of a new meeting example.

    Each ActionHandler has tags, which consist of a list of the events that could be processed by it.
    """

    @abstractmethod
    def start(self, example_id: str, scenario_ids: List[str], base_examples: List[Dict], scenario_id_index: int,
              env_state: Optional[Any] = None, **kwargs) -> Tuple[bool, Dict, int, Any]:
        """
        Starts the sampling for a new meeting and return the first sampled source, together with an offset.
        The offset should be a non-negative integer.

        Args:
            example_id: ID of the current meeting for which a start example should be sampled
            scenario_ids: IDs of the scenarios/speakers which should be in the generated meeting
            base_examples: List of base examples, each scenario/speaker has one according base example
            scenario_id_index: Index of the starting speaker/scenario with respect to the list of base examples
            env_state: (optional) State of the environment, can be used to provide additional information
                       to the ActionHandler

        Returns: Tuple with four entries:
                    - Boolean that indicates the success of the start
                    - Sampled source
                    - offset
                    - potentially changed environment state
        """
        raise NotImplementedError

    @abstractmethod
    def next_scenario(self, action: Any, scenario_id_index: int, examples: List[Dict], base_examples: List[Dict],
                      env_state: Optional[Any] = None, **kwargs) -> Tuple[bool, Optional[Dict], int, Any]:
        """
        Samples a fitting source and offset for the given action and speaker.

        Args:
            action: Action that determines the transition between the last speaker and the current speaker
            scenario_id_index: Index of the current speaker/scenario with respect to the list of base examples
            examples: A list of source that the newly generated meeting example contains to this point
            base_examples: List of base examples, each scenario/speaker has one according base example
            env_state: (optional) State of the environment, can be used to provide additional information
                to the ActionHandler

        Returns: Tuple with four entries:
                    - Boolean that indicates the success of the Action
                    - Sampled source, when available
                    - offset
                    - potentially changed environment state
        """
        raise NotImplementedError

    @abstractmethod
    def set_datasets(self, normalized_datasets: Dict[str,  Dataset], use_vad: bool = False) -> None:
        """
        Sets the datasets from which the sources for the corresponding actions are sampled.
        First a specify dataset is used, otherwise the default dataset ('*' as key).

        Args:
            normalized_datasets: Dictionary which maps actions to normalized datasets (keys: action, item: dataset)
                Datasets can be normalized by calling mms_msg.sampling.utils.cache_and_normalize_input_dataset
            use_vad: (optional) Is VAD data in the given datasets and should these data be used during
                the selection of samples and offset. Default Value: False
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def tags(self) -> Set[str]:
        """
        Tags of this class. The tags contain information about which types of action
            the ActionHandler can process. A '*' means that this handler can process any type of action.

        Returns: Set with all tags
        """
        raise NotImplementedError


class DistributionActionHandler(ActionHandler):
    """
    Action Handler that can handle four actions: (TH: Turn Hold, TS: Turn Switch, OV: Overlap, BC: Backchannel).
    For each action a source with a fitting offset is sampled.
    The offset is determined by first computing some values that depends on the action
    (TH -> silence, TS -> silence, OV -> overlap, BC -> backchannel offset)
    For all three of these values (silence, overlap, backchannel offset) a sampler with some distribution is used,
    which depends on the according sampler.
    When the offset is calculated from these intermediate values then VAD data is also taken into account,
    when available.

    Important! Due to the internal selection of the samples the distribution of the offsets depends
    on the given input dataset. When using the statistics from a dataset are used and then this dataset is used as
    input dataset the mean of the resulting overlap distribution is typically smaller than the original.
    This effect increases when the process of sampling and generation is done multiple times.
    Thus, this is not recommended.

    Properties:
        overlap_sampler: Used sampler for the overlap
        silence_sampler: Used sampler for the silence
        backchannel_start_sampler: Used sampler for offset off the backchannel source
        border_margin:  Used as minimal overlap during the OV action and minimal spacing
            of the backchannel source from the borders of the foreground source.
        use_vad: Is VAD data present in the given datasets and should this data be used for determining sources
                 and offsets.

        scenario_ids: (read only) Scenario ids of the used speakers in the currently processed example
        example_id: (read only) ID of the currently processed example
        last_foreground_speaker: (read only) Scenario ID of the speaker that is currently active in the foreground
        base_examples: (read only) Base examples for the scenarios of all used speakers of the active example
        grouped_datasets: (read only) Dictionary with the datasets that are used to sample for each action.
            The datasets themselves are grouped by scenario_id. The actions are the keys,
            while the datasets are the values in the dictionary. '*' marks default dataset.
    """
    def __init__(self, overlap_sampler: OverlapSampler, silence_sampler: SilenceSampler,
                 backchannel_start_sampler: BackchannelStartSampler = BackchannelStartSampler(),
                 bc_border_margin: int = 100):
        """
        Initialization of the action handler with samplers.
        Important: After the initialization you have to provide datasets via the set_dataset function

        Args:
            overlap_sampler: Used sampler for the overlap
            silence_sampler: Used sampler for the silence
            backchannel_start_sampler: Used sampler for offset off the backchannel source
            bc_border_margin: (optional) Used as minimal spacing of the backchannel source
                from the borders of the foreground source.
        """

        self.overlap_sampler = overlap_sampler
        self.silence_sampler = silence_sampler
        self.backchannel_start_sampler = backchannel_start_sampler
        self.bc_border_margin = bc_border_margin
        self.use_vad = None

        self._scenario_ids = None
        self._example_id = None

        self._last_foreground_speaker = None

        self._base_examples = None
        self._grouped_datasets = None

    def set_datasets(self, normalized_datasets: Dict[str, Dataset], use_vad: bool = False) -> None:
        self._grouped_datasets = {key: dataset.groupby(lambda x: x['scenario']) for
                                                      (key, dataset) in normalized_datasets.items()}
        self.use_vad = use_vad

        if not ('*' in self._grouped_datasets.keys()
                or self.tags.issubset(self._grouped_datasets.keys())):
            raise AssertionError(("The Tags of the normalized datasets and the ActionHandler do not fit, some actions "
                                  "have no fitting dataset and thus no fitting sources can be sampled"
                                  " for these actions. You can set a dataset as default that is used"
                                  " when no specific dataset is available by using '*' as key."),
                                 "Missing Tags: ", self.tags.difference(self._grouped_datasets.keys()))

    def start(self, example_id: str, scenario_ids: List[str], base_examples: List[Dict], scenario_id_index: int,
              env_state: Optional[Any] = None, **kwargs) -> Tuple[bool, Dict[str, Any], int, Any]:
        self._scenario_ids = scenario_ids
        self._example_id = example_id

        # Adding the first speaker
        current_source = copy.deepcopy(base_examples[scenario_id_index])
        offset = 0
        self._last_foreground_speaker = scenario_ids[scenario_id_index]

        return True, current_source, offset, None

    def next_scenario(self, action: Any, scenario_id_index: int, examples: List[Dict], base_examples: List[Dict],
                      env_state: Optional[Any] = None, **kwargs) -> Tuple[bool, Optional[Dict[str, Any]], int, Any]:

        assert self._grouped_datasets is not None, \
            "set_datasets has to be called, before using the ActionHandler"

        current_scenario = self._scenario_ids[scenario_id_index]
        segment_idx = len(examples)

        offset = 0
        current_source = None

        if self.use_vad:
            source_key = 'aligned_source'
            source_key2 = 'aligned_source'
        else:
            source_key = 'original_source'
            source_key2 = 'observation'

        # Select the fitting dataset for the current action, when for this action no dataset is available,
        # the default dataset is chosen
        if action in self._grouped_datasets.keys():
            current_dataset = self._grouped_datasets[action]
        else:
            current_dataset = self._grouped_datasets["*"]

        # Determine fitting source and offset
        try:
            if action in ("TH", "TS"):
                current_source, offset = self._action_th_ts(current_scenario, current_dataset, examples, segment_idx,
                                                            source_key)
            elif action == "OV":
                current_source, offset = self._action_ov(current_scenario, current_dataset, examples, segment_idx,
                                                         source_key)
            elif action == "BC":
                current_source, offset = self._action_bc(current_scenario, current_dataset, examples, segment_idx,
                                                         source_key, source_key2)
        except ValueError:
            # Sampling of Offset failed
            return False, None, -1, None

        if not (current_source is None):
            if self.use_vad:
                offset = offset - current_source['offset']['aligned_source']

            # Prevent negative offsets
            offset = max(0, offset)

            return True, current_source, offset, None
        else:
            return False, None, -1, None

    def _sample_source(self, current_scenario: str, current_dataset: Union[Dict, Dataset], examples: List[Dict],
                       segment_idx: int) -> Dict[str, Any]:
        """
        Internal function that samples a source from the current scenario from the current dataset using
        the random round-robin method. The archive consistency for multiple executions all previously
        sampled examples and the index of the current exampled are used as seed for the random number generator.

        Args:
            current_scenario: Scenario from which the source should be sampled
            current_dataset: Dataset from which the source should be sampled
            examples: List of previously sampled sources (used as seed for rng)
            segment_idx: Index of the currently sampled source (used as seed for rng)

        Returns: Dictionary which represents the sampled source
        """

        current_source_id = sequence_sampling.sample_random_round_robin(
            current_dataset[current_scenario].keys(),
            sequence=[x['example_id'] for x in examples if x['scenario'] == current_scenario],
            rng=get_rng(self._example_id, 'example', segment_idx),
        )
        current_source = copy.deepcopy(current_dataset[current_scenario][current_source_id])
        return current_source

    def _action_th_ts(self, current_scenario: str, current_dataset: Union[Dict, Dataset], examples: List[Dict],
                      segment_idx: int, source_key: str) -> Tuple[Dict[str, Any], int]:
        """
        Internal function for handling the Turn hold (TH) and Turn switch (TS) actions.
        Samples a fitting source from the given dataset, computes the offset of the sampled source and updates
        the current active foreground speaker.

        Args:
            current_scenario: Scenario from which the source should be sampled
            current_dataset: Dataset from which the source should be sampled
            examples: List of previously sampled sources (used as seed for rng)
            segment_idx: Index of the currently sampled source (used as seed for rng)
            source_key: Key for accessing the values in the dictionary which represent single sources.
                Depends on the usage of VAD data (No VAD: original_source, VAD: aligned_source)

        Returns: Tuple of the sampled source and the corresponding offset
        """

        current_source = self._sample_source(current_scenario, current_dataset, examples, segment_idx)
        silence = self.silence_sampler(get_rng(self._example_id, segment_idx, 'silence'))
        self._last_foreground_speaker = current_scenario
        offset = max([x['speaker_end'][source_key] for x in examples]) + silence

        return current_source, offset

    def _action_ov(self, current_scenario: str, current_dataset: Union[Dict, Dataset], examples: List[Dict],
                   segment_idx: int, source_key: str) -> Tuple[Dict[str, Any], int]:
        """
        Internal function for handling the Overlap (OV) action.
        Samples a fitting source from the given dataset, computes the offset of the sampled source and updates
        the current active foreground speaker.

        Args:
            current_scenario: Scenario from which the source should be sampled
            current_dataset: Dataset from which the source should be sampled
            examples: List of previously sampled sources (used as seed for rng)
            segment_idx: Index of the currently sampled source
            source_key: Key for accessing the values in the dictionary which represent single sources.
                Depends on the usage of VAD data (No VAD: original_source, VAD: aligned_source)

        Returns: Tuple of the sampled source and the corresponding offset
        """

        current_source = self._sample_source(current_scenario, current_dataset, examples, segment_idx)
        overlap = self.overlap_sampler(examples, current_source,
                                       rng=get_rng(self._example_id, segment_idx, 'overlap'),
                                       use_vad=self.use_vad)
        self._last_foreground_speaker = current_scenario

        offset = max([x['speaker_end'][source_key] for x in examples]) - overlap

        return current_source, offset

    def _action_bc(self, current_scenario: str, current_dataset: Union[Dict, Dataset], examples: List[Dict],
                   segment_idx: int, source_key: str, source_key2: str) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Internal function for handling the Backchannel action (BC).
        Samples a fitting source from the given dataset and computes the offset of the sampled source.

        Args:
            current_scenario: Scenario from which the source should be sampled
            current_dataset: Dataset from which the source should be sampled
            examples: List of previously sampled sources (used as seed for rng)
            segment_idx: Index of the currently sampled source
            source_key: Key for accessing the values in the dictionary which represent single sources.
                           Depends on the usage of VAD data (No VAD: original_source, VAD: aligned_source)
            source_key2: Second Key for accessing the values in the dictionary which represent single sources.
                            Depends on the usage of VAD data (No VAD: observation, VAD: aligned_source)

        Returns: Tuple of the sampled source and the corresponding offset
        """

        last_foreground_example = list(filter(lambda x: x['speaker_id'] == self._last_foreground_speaker, examples))[-1]

        backchannel_speaker_ends = [x['speaker_end'] for x in
                                    list(filter(lambda x: not x['speaker_id'] == self._last_foreground_speaker,
                                                examples))]

        foreground_length = last_foreground_example['num_samples'][source_key]
        free_backchannel_length = last_foreground_example['speaker_end'][source_key] - max(
            [x[source_key] for x in backchannel_speaker_ends] + [0])

        max_allowed_length = min(foreground_length, free_backchannel_length) - 2 * self.bc_border_margin

        # Rejection sampling of the backchannel source
        current_source = rejection_sampling(get_rng(self._example_id, 'example', segment_idx), current_scenario,
                                            current_dataset, examples, max_length=max_allowed_length)
        current_source = copy.deepcopy(current_source)

        if current_source is not None:
            min_possible_start_offset = max(max([x[source_key] for x in backchannel_speaker_ends] + [0]),
                                            last_foreground_example['offset'][source_key]) \
                                        + self.bc_border_margin
            max_possible_start_offset = last_foreground_example['speaker_end'][source_key] - \
                current_source['num_samples'][source_key2] - self.bc_border_margin

            offset = self.backchannel_start_sampler(min_possible_start_offset, max_possible_start_offset,
                                                    get_rng(self._example_id, segment_idx, 'start_offset'))
            return current_source, offset
        else:
            logger.warning("No fitting backchannel source found.")
            return None, 0

    @property
    def tags(self) -> Set[str]:
        return {"TS", "TH", "OV", "BC"}

    @property
    def scenario_ids(self) -> List[str]:
        return self._scenario_ids

    @property
    def example_id(self) -> str:
        return self._example_id

    @property
    def last_foreground_speaker(self) -> str:
        return self._last_foreground_speaker

    @property
    def grouped_datasets(self) -> Dict[str, Union[Dict, Dataset]]:
        return self._grouped_datasets


def rejection_sampling(rng: np.random.Generator, current_scenario: str, current_dataset: Union[Dict, Dataset],
                       examples: List[Dict], max_tries: int = 100, min_length: int = 0,
                       max_length: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Uses rejection sampling to get a source that has more than min and less than max samples.
    When no fitting sample can be found, None is returned.

    Args:
        rng: random number generator that is used for the sampling
        current_scenario: scenario from which the source should be sampled
        current_dataset: dataset from which the source should be sampled
        examples: list of sources that were sampled until now
        max_tries: maximum amount of tries, when after these amount of tries no fitting source is found,
            None is returned
        min_length: minimal amount of samples that the source should have
        max_length: maximal amount of samples that the source should have

    Returns: source: when its fitting, None: when no fitting source is found
    """

    for tries in range(max_tries):
        current_source_id = sequence_sampling.sample_random_round_robin(
            current_dataset[current_scenario].keys(),
            sequence=[
                x['example_id'] for x in examples
                if x['scenario'] == current_scenario],
            rng=rng
        )
        current_source = copy.deepcopy(current_dataset[current_scenario][current_source_id])

        if current_source['num_samples']['observation'] >= min_length and (
                max_length is None or current_source['num_samples']['observation'] <= max_length):
            return current_source
    # When no fitting source is found None is returned
    return None
