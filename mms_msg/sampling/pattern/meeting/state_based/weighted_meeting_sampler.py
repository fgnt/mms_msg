import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict

from cached_property import cached_property

from mms_msg import keys
import padertorch as pt
import paderbox as pb
from lazy_dataset import Dataset, FilterException, from_dict
from mms_msg.sampling.utils import update_num_samples, cache_and_normalize_input_dataset, collate_fn
from mms_msg.sampling.utils.rng import get_rng_example

from mms_msg.sampling.pattern.meeting.state_based.transition_model import SpeakerTransitionModel
from mms_msg.sampling.pattern.meeting.state_based.action_handler import ActionHandler

logger = logging.getLogger('meeting_generation')


class _WeightedMeetingSampler:
    """
    Main class of the WeightedMeetingSampler.

    The WeighedMeetingSampler requires two internal components to work:
    - A TransitionModel which determines the sequence of the speakers and transition types between them (actions).
    - An ActionHandler which processes the action by sampling a source and determines a fitting offset

    TransitionModel and ActionHandler are abstract classes. Due to this it is possible to customize the behavior
    of the meeting generator to a great extent. In the files transition_model.py and action_handler.py
    possible example implementations for these classes can be found, that are also used in meeting_generator.py

    During generation a new meeting the WeighedMeetingSampler runs through the following loop,
    until the desired length of the meeting is reached or an error during the generation happens.
    - The transition model outputs next action and speaker
    - The action handler processes these two values and samples a fitting source of the speaker
      together with a fitting offset
    - The keys of the selected source are adjusted to the sampled offset

    When the generation of a meeting fails, a FilterException is raised.

    Properties:
        input_datasets: Dictionary of datasets that are used for generation of new dialogue examples,
            for each dataset an action can be specified, when this action is selected the following sample
            is drawn from the corresponding dataset. When this property is set the normalized_datasets property
            is also adopted to the new input datasets.
        normalized_datasets: (read only) dictionary of the normalized input datasets
        duration: Duration that the newly generated examples should roughly have, can be slightly exceeded
        transition_model: Transition-model that determines the sequence of speaker and the transitions (actions)
            between them.
        action_handler: Action handler that is responsible for processing the transitions from the transition model.
            Draws fitting samples from the input dataset for each action.
        use_vad: Should VAD data be used, only activate when VAD exists in the input dataset
    """

    def __init__(self, input_datasets: [Dataset, Dict], transition_model: SpeakerTransitionModel,
                 action_handler: ActionHandler, duration: Optional[int] = 960000, use_vad: Optional[bool] = False,
                 force_fitting_tags: Optional[bool] = True):
        """
        Initializes the WeightedMeetingSampler with a TransitionModel and an ActionHandler.
        Also, the input datasets for the different actions are set.

        Args:
            input_datasets: Dictionary of datasets that are used for generation of new dialogue examples,
                for each dataset an action can be specified, when this action is selected the following sample
                is drawn from the corresponding dataset.
            transition_model: Transition-model that determines the sequence of speaker and the transitions (actions)
                between them.
            action_handler: Action handler that is responsible for processing the transitions from the transition model.
                Draws fitting samples from the input dataset for each transition.
            duration: Duration that the newly generated examples should roughly have, can be slightly exceeded
            use_vad: Should VAD data be used, only activate when VAD exists in the input dataset
            force_fitting_tags: When set to True, the class enforces that all actions that can be generated
                from the transition model can be processed by the action handler, by comparing their tags.
        """
        self._input_datasets = dict()
        for k, v in input_datasets.items():
            # Transforms datasets in the shape of a dictionary to the Type Dataset
            if type(v) is dict:
                v = from_dict(v)
            # Remove FilterExceptions
            self._input_datasets[k] = v.catch()

        self.duration = duration
        self.transition_model = transition_model
        self.action_handler = action_handler
        self.action_handler.set_datasets(self.normalized_datasets, use_vad)
        self.use_vad = use_vad

        if force_fitting_tags:
            # Check if arbitrary action scan be handled by the action handler (*)
            # or all possible actions of the transition model can be handled (subset relation)
            if not ('*' in self.action_handler.tags
                    or self.transition_model.tags.issubset(self.action_handler.tags)):
                raise AssertionError(("The Tags of the TransitionModel and the ActionHandler do not fit, some actions "
                                      "from the TransitionModel can't be handled by the ActionHandler. "
                                      "To disable the enforcement of fitting tags set force_fitting_tracks to False."),
                                     "Not supported Tags: ",
                                     self.transition_model.tags.difference(self.action_handler.tags))

    @property
    def input_datasets(self):
        return self._input_datasets

    @input_datasets.setter
    def input_datasets(self, input_datasets):
        # Invalidate cached property that is calculated from input_dataset
        delattr(self, "normalized_datasets")

        self._input_datasets = dict()
        for k, v in input_datasets.items():
            # Transforms datasets in the shape of a dictionary to the Type Dataset
            if type(v) is dict:
                v = from_dict(v)
            # Remove FilterExceptions
            self._input_datasets[k] = v.catch()

        # Update ActionHandler with new datasets
        self.action_handler.set_datasets(self.normalized_datasets, self.use_vad)

    @cached_property
    def normalized_datasets(self) -> Dict[str, Union[Dataset, Dict]]:
        return {key: cache_and_normalize_input_dataset(dataset) for (key, dataset) in self._input_datasets.items()}

    def _log_action(self, i: int, action: [Any], current_source: Dict[str, Any]) -> None:
        """
        Internal function that logs the action and corresponding source. Logged keys of the source: scenario, offset,
        speaker_end. When VAD is used the aligned source values are logged, otherwise the original_source is used.

        Args:
            i: index of the action in the final output
            action: action that should be logged
            current_source: current active source that should be logged
        """

        key = 'original_source'
        if self.use_vad:
            key = 'aligned_source'

        logger.info("i: %s, action: %s, scenario: %s, offset: %s, speaker_end: %s", i, action,
                    current_source['scenario'],
                    current_source['offset'][key], current_source['speaker_end'][key])

    def _set_keys(self, source: Dict[str, Any], offset: int) -> None:
        """
        Internal function that sets the offset, num_samples and speaker_end keys for a source with a certain offset.
        Additional keys are set when a vad is used in the generation process.
        The keys are set in-place, so the source used as input is changed.

        Args:
            source: source for which the keys are set
            offset: desired offset of the source, offset must be the offset
                for the original_source not the aligned_source
        """

        # Aligned source: Speaker active

        for key in ['offset', 'num_samples', 'speaker_end']:
            if key not in source.keys():
                source[key] = dict()

        if self.use_vad:
            source['offset']['aligned_source'] = offset + source['offset']['aligned_source']
            source['speaker_end']['aligned_source'] = (source['offset']['aligned_source']
                                                       + source['num_samples']['aligned_source'])

        source['offset']['original_source'] = offset
        source['num_samples']['original_source'] = source['num_samples']['observation']
        source['speaker_end']['original_source'] = offset + source['num_samples']['original_source']

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Used for generating of a meeting, with transitions according to the given transition model
        and sources and offsets according to the given action handler.
        The generation is based on base examples which select the active speakers.

        Args:
            example: List of base example, which determine which speakers are active in the generated meeting

        Returns: Meeting generated according to the transition model and the action handler.
        """

        example_id = example['example_id']

        if 'ST' in self.normalized_datasets.keys():
            base_examples = [self.normalized_datasets['ST'][source_id] for source_id in example['source_id']]
        else:
            base_examples = [self.normalized_datasets['*'][source_id] for source_id in example['source_id']]

        logger.info(f'Generating meeting with example ID "{example_id}"')

        # Sample stuff that is constant over the meeting
        scenario_ids = [x['scenario'] for x in base_examples]

        # A generated example is valid, if every chosen speaker has at least one sample
        valid = False

        examples = []

        transition_rng = get_rng_example(example, 'transition')

        # Get first speaker
        scenario_id_index, state = self.transition_model.start()
        success, current_source, offset, state = self.action_handler.start(example_id, scenario_ids, base_examples,
                                                                           scenario_id_index, env_state=state)
        if success:
            self._set_keys(current_source, offset)
            self._log_action(0, "ST", current_source)
            examples.append(current_source)
            valid = True
        else:
            logger.error("Start action failed.")

        # Add samples until the length overshoots the desired length, then stop
        while valid and (max([example[keys.OFFSET][keys.ORIGINAL_SOURCE] + example['num_samples']['observation']
                              for example in examples], default=0) < self.duration):

            try:
                action, scenario_id_index, state = self.transition_model.next(rng=transition_rng,
                                                                              last_action_success=success,
                                                                              env_state=state, examples=examples)

                success, current_source, offset, state = self.action_handler.next_scenario(action, scenario_id_index,
                                                                                           examples, base_examples,
                                                                                           env_state=state)

                if success:
                    self._set_keys(current_source, offset)
                    self._log_action(len(examples), action, current_source)
                    examples.append(current_source)
                else:
                    logger.warning(f"Can not find fitting source for Action {action}, retry.")

            except StopIteration:
                # No valid action possible.
                valid = False
                logger.error("No valid action possible.")

        # Check if each speaker has at least one appearance
        scenarios = {scenario_example['scenario'] for scenario_example in examples}

        if scenarios != set(scenario_ids):
            logger.error(f'The speakers present in the meeting, do not correspond to those in the base examples.'
                         f' Missing scenarios: ' + str(set(scenario_ids).difference(scenarios)).replace('set()', '{}') +
                         f' Surplus scenarios: ' + str(scenarios.difference(set(scenario_ids))).replace('set()', '{}'))
            valid = False

        # Generation not successful when the first state cannot be initialized, no valid action can be found
        # or the present speakers do not fit the base examples
        if not valid:
            logger.error('Generation not successful for: %s', example_id)
            # When the generation of an example is not successful a filter exception is raised
            raise FilterException()

        # Collate the examples and copy over / replicate things that are already
        # present in the base full overlap example.
        # Use the same format as SMS-WSJ.
        # Heuristic: Replicate nothing that is in the collated
        # example. For the rest, we have a white- and blacklist of keys that should
        # or should not be replicated. Keys that are not replicated are copied from
        # the input example to the output example.

        replicate_whitelist = (
            'log_weights',
            'audio_path.rir',
            'source_position',
        )
        replicate_blacklist = (
            'sensor_positions',
            'room_dimensions',
            'example_id',
            'num_speakers',
            'source_dataset',
            'sound_decay_time',
            'sensor_position',
            'snr',
        )

        # Collate
        collated_examples = collate_fn(examples)

        # Handle some special keys prior to replication
        collated_examples['source_id'] = collated_examples.pop('example_id')
        flat_example = pb.utils.nested.flatten(example)
        speaker_ids = flat_example['speaker_id']
        collated_examples['num_samples'].pop('observation')

        sources = collated_examples['audio_path'].pop('observation')
        collated_examples['audio_path']['original_source'] = sources
        collated_examples['source_dataset'] = collated_examples['dataset']
        update_num_samples(collated_examples)
        collated_examples['dataset'] = example['dataset']

        # Copy and replicate
        flat_collated_example = pb.utils.nested.flatten(collated_examples)
        for key in flat_example.keys():
            if key not in flat_collated_example:
                if key in replicate_whitelist:
                    if key == 'source_position':
                        # Special case: nested lists
                        assert len(flat_example[key][0]) == len(speaker_ids), (flat_example[key], speaker_ids)
                        transposed = zip(*flat_example[key])
                        m = dict(zip(speaker_ids, transposed))
                        transposed = [m[s] for s in flat_collated_example['speaker_id']]
                        flat_collated_example[key] = list(map(list, zip(*transposed)))
                    else:
                        assert len(flat_example[key]) == len(speaker_ids), (flat_example[key], speaker_ids)
                        m = dict(zip(speaker_ids, flat_example[key]))
                        flat_collated_example[key] = [m[s] for s in flat_collated_example['speaker_id']]
                else:
                    if key not in replicate_blacklist:
                        # Add keys that you need to blacklist/whitelist
                        raise RuntimeError(
                            f'Key {key} not found in replicate_whitelist or '
                            f'replicate_blacklist.\n'
                            f'replicate whitelist={replicate_whitelist},\n'
                            f'replicate whitelist={replicate_blacklist},\n'
                        )
                    flat_collated_example[key] = flat_example[key]
        collated_example = pb.utils.nested.deflatten(flat_collated_example)
        return collated_example


@dataclass(frozen=True)
class WeightedMeetingSampler(pt.Configurable):
    """
    Wrapper class for _WeightedMeetingSampler that
    Samples a meeting from (full-overlap) base examples.

    Properties:
        transition_model: Transition-model that determines the sequence of speaker and the transitions (actions)
            between them.
        action_handler: Action handler that is responsible for processing the transitions from the transition model.
            Draws fitting samples from the input dataset for each action.
        duration: Duration that the newly generated examples should roughly have, can be slightly exceeded
        use_vad: Should VAD data be used, only activate when VAD exists in the input dataset
        force_fitting_tags: When set to True, the class enforces that all actions that can be generated
            from the transition model can be processed by the action handler, by comparing their tags.
    """

    transition_model: SpeakerTransitionModel
    action_handler: ActionHandler
    duration: int = 120 * 8000
    use_vad: bool = False
    force_fitting_tags: bool = True

    def __call__(self, input_datasets: Dict[str, Union[Dict, Dataset]]) -> _WeightedMeetingSampler:
        """
        Initialises the _WeightedMeetingSampler with input datasets and uses
        the values of the dataclass for the other parameters.

        Args:
        input_datasets: Dictionary of datasets that are used for generation of new dialogue examples,
            for each dataset an action can be specified, when this action is selected the following sample
            is drawn from the corresponding dataset.

            The examples in the datasets must contain the following keys:
                - scenario (string): Only utterances with the same scenario are put
                    into the same meeting for the same speaker. Example: In LibriSpeech,
                    the environment changes heavily for different chapters, even if
                    the speaker stays the same. So, we want the chapter to stay the same
                    during one meeting.
                - num_samples (int): length of audio signal
                - (optional) vad (ArrayInterval or numpy array): VAD information, with sample
                    resolution

        Returns: WeightedMeetingSampler that is initialised with the given input datasets
        """
        return _WeightedMeetingSampler(
            input_datasets=input_datasets,
            duration=self.duration,
            transition_model=self.transition_model,
            action_handler=self.action_handler,
            use_vad=self.use_vad,
            force_fitting_tags=self.force_fitting_tags
        )
