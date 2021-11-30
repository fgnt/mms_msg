import copy
import functools
import logging
import operator
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from padercontrib.database import keys

import paderbox as pb
from lazy_dataset import Dataset
from .overlap_sampler import OverlapSampler, UniformOverlapSampler
from .scenario_sequence_sampler import sample_balanced
from .. import update_num_samples, normalize_example, cache_and_normalize_input_dataset
from ..utils import collate_fn, get_rng_example
from ..utils.sampling import sample_random_round_robin

logger = logging.getLogger('meeting')


def sample_meeting_from_full_overlap(
        example, input_dataset,
        *,
        duration: int = 600 * 8000,
        overlap_sampler: OverlapSampler = UniformOverlapSampler(
            max_concurrent_spk=2,
            p_silence=0.1,
            maximum_silence=16000,
            maximum_overlap=40000
        ),
        scenario_sequence_sampler: callable = sample_balanced,
):
    input_dataset = cache_and_normalize_input_dataset(input_dataset)
    example_id = example['example_id']
    base_examples = [input_dataset[source_id] for source_id in example['source_id']]
    logger.debug(f'Generating meeting with example ID "{example_id}"')

    # We need to sample explicitly from a scenario, so group by it
    grouped_dataset = input_dataset.groupby(lambda x: x['scenario'])

    # Sample stuff that is constant over the meeting
    scenario_ids = [x['scenario'] for x in base_examples]
    scenario_sequence_rng = get_rng_example(example, 'speaker_sequence')

    examples = []

    # Add base examples to be sure that each speaker is active at least once
    while (
            max([
                example[keys.OFFSET][keys.ORIGINAL_SOURCE] +
                example['num_samples']['observation'] for example in examples
            ], default=0) < duration
            or len(base_examples) > 0
    ):
        segment_idx = len(examples)
        # Get the input example to add to the meeting. First, add each
        # base example once, then sample randomly according to the strategy
        # defined by the scenario sampler and random choice
        if len(base_examples) > 0:
            current_source = base_examples.pop(0)
            current_scenario = current_source['scenario']
            logger.debug(f'Sampling base example "{current_scenario}"')
        else:
            current_scenario = scenario_sequence_sampler(
                scenario_ids, examples, scenario_sequence_rng
            )
            logger.debug(f'Sampling for scenario "{current_scenario}"')

            current_source_id = sample_random_round_robin(
                grouped_dataset[current_scenario].keys(),
                sequence=[
                    x['example_id'] for x in examples
                    if x['scenario'] == current_scenario
                ],
                rng=get_rng_example(example, 'example', segment_idx),
            )
            current_source = input_dataset[current_source_id]
        current_source = copy.copy(current_source)

        # Sample either overlap or silence duration
        rng = get_rng_example(example, segment_idx, 'offset')
        if len(examples) > 0:
            offset = overlap_sampler(examples, current_source, rng)
        else:
            offset = 0
        current_source[keys.OFFSET] = {keys.ORIGINAL_SOURCE: offset}
        current_source['speaker_end'] = offset + current_source['num_samples']['observation']

        examples.append(current_source)

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
    )

    # Collate
    collated_examples = collate_fn(examples)

    # Handle some special keys prior to replication
    collated_examples['source_id'] = collated_examples.pop('example_id')
    flat_example = pb.utils.nested.flatten(example)
    speaker_ids = flat_example['speaker_id']
    collated_examples['num_samples'] = {
        'original_source': collated_examples['num_samples']['observation']
    }
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
class MeetingSampler:
    """
    Wrapper class around `sample_meeting_examples` for easier use with
    `pt.Configurable`. Calling this samples one meeting with the given settings.

    Sampling is deterministic based on the example_id and the contents of the
    input `dataset`.
    """
    duration: int = 600
    scenario_sequence_sampler: callable = sample_balanced
    overlap_sampler: OverlapSampler = field(
        default_factory=functools.partial(
            UniformOverlapSampler,
            max_concurrent_spk=2,
            p_silence=0.1,
            maximum_silence=16000,
            maximum_overlap=40000,
        )
    )

    def __call__(self, dataset: Dataset):
        """
        The examples in the grouped dataset must contain the following keys:
         - scenario (string): Only utterances with the same scenario are put
            into the same meeting for the same speaker. Example: In LibriSpeech,
            the environment changes heavily for different chapters, even if
            the speaker stays the same. So, we want the chapter to stay the same
            during one meeting.
         - num_samples (int): length of audio signal
         - vad (ArrayInterval or numpy array): VAD information, with sample
            resolution
        """
        return functools.partial(
            sample_meeting_from_full_overlap,
            input_dataset=dataset,
            duration=self.duration,
            scenario_sequence_sampler=self.scenario_sequence_sampler,
            overlap_sampler=self.overlap_sampler,
        )
