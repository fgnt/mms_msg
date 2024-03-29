import operator
from functools import partial

import lazy_dataset

import numpy as np
from sacred import Experiment

from lazy_dataset.database import JsonDatabase
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.sampling.utils.utils import collate_fn
from mms_msg.sampling.utils.rng import get_rng
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.pattern.meeting.meeting_sampler import MeetingSampler
import paderbox as pb
from .wsj_meeting import OVERLAP_SETTINGS
from ..reverberation.sms_wsj import SMSWSJRIRDatabase

ex = Experiment('mixture_generator_create_json')


@ex.config
def defaults():
    source_json_path = ''
    rir_json_path = ''
    assert len(source_json_path) > 0, 'Path to WSJ JSON needs to be set for test meeting generation!'
    json_path = 'test_meetings_wsj8k.json'
    meeting_duration = 120 * 8000
    num_speakers = (5, 6, 7, 8)
    overlap_conditions = dict(no_ov=OVERLAP_SETTINGS['no_ov'], medium_overlap=OVERLAP_SETTINGS['medium_ov'],
                              high_overlap=OVERLAP_SETTINGS['high_ov'])
    dataset_name = "test_eval92"
    reverb = True
    if reverb:
        assert len(rir_json_path) > 0, 'RIR JSON path needs to be specified for the generation of reverberated meetings.' \
                                       'Either turn of reverb or execute "mms_msg.databases.reverberation." ' \
                                       'to generate the required RIRs.'
    n_meetings = 16  # Number of meetings per #spks


def _speaker_composition_list_to_dict(
        composition: list,
        dataset_name: str,
) -> dict:
    base = {}
    for idx, composition in enumerate(composition):
        # Combine the sampled examples to one multi-speaker example with a
        # format similar to SMS-WSJ
        example = collate_fn(composition)
        example['num_speakers'] = len(example['speaker_id'])
        example['source_dataset'] = example['dataset']

        # The new dataset name is a combination of the given dataset name and
        # the dataset name of the base example. This only works if all examples
        # come from the same source dataset
        assert pb.utils.misc.all_equal(example['source_dataset']), (
            'Dataset name is not equal! Implement something new.'
        )
        example['dataset'] = dataset_name

        # Move audio_path.observation and num_samples to 'original_source' to
        # match SMS-WSJ and to make room for additional keys in the audio_path
        # and num_samples sub-dicts
        example['audio_path'] = {
            'original_source': example['audio_path'].pop('observation')
        }
        example['num_samples'] = {
            'original_source': pb.utils.nested.get_by_path(
                example, 'num_samples.observation'
            )
        }

        # Check that there are no duplicate speakers
        assert pb.utils.misc.all_unique(example['speaker_id']), example['speaker_id']

        # Build an example ID for each example
        example_id = '_'.join([str(idx), *map(str, example['example_id'])])
        assert example_id not in base, (
            'Duplicate example IDs found! Modify the example ID generation '
            'code to avoid this!'
        )
        example['source_id'] = example['example_id']
        example['example_id'] = example_id

        base[example_id] = example
    return base


def sample_rirs_for_test_set(example, rir_dataset, rng):

    rir_example = rir_dataset.random_choice(rng_state=rng)
    rirs = rir_example['audio_path']['rir']
    num_speakers = len(example['speaker_id'])
    assert num_speakers <= len(rirs), (num_speakers, len(rirs))
    example['audio_path']['rir'] = rirs[:num_speakers]
    example['room_dimensions'] = rir_example['room_dimensions']
    example['sound_decay_time'] = rir_example['sound_decay_time']
    example['sensor_position'] = rir_example['sensor_position']
    example['source_position'] = [s[:num_speakers] for s in rir_example['source_position']]
    return example


@ex.automain
def main(json_path, overlap_conditions, meeting_duration, source_json_path, rir_json_path,
         reverb, num_speakers, dataset_name, n_meetings, _log):
    database = JsonDatabase(source_json_path)

    database_dict = {
        'datasets': {
        },
        'meta': {
        }
    }
    test_dataset = database.get_dataset(dataset_name)
    test_dataset_per_spk = test_dataset.groupby(lambda x: x['speaker_id'])
    available_spk = list(sorted(test_dataset.groupby(lambda x: x['speaker_id']).keys()))
    rng = get_rng(dataset_name)
    database_dict['datasets'] = {scenario: {} for scenario in overlap_conditions.keys()}

    assert len(available_spk) >= max(num_speakers)
    for n_spk in num_speakers:
        speaker_constellations = []
        start_examples = []
        idx_constellation = [rng.permutation(len(available_spk)) for i in range(n_meetings)]
        for constellation in idx_constellation:
            constellation = constellation[:n_spk]
            speaker_constellations.append(np.array([available_spk[c] for c in constellation]))
        for speaker_constellation in speaker_constellations:
            start_examples.append([test_dataset_per_spk[spk].random_choice(rng_state=rng) for spk in speaker_constellation])

        for scenario, overlap_sampler in overlap_conditions.items():
            overlap_sampler_class = UniformOverlapSampler(**overlap_sampler)


            meeting_sampler = MeetingSampler(duration=meeting_duration, overlap_sampler=overlap_sampler_class)
            meeting_dataset = lazy_dataset.new(_speaker_composition_list_to_dict(start_examples, dataset_name))
            meeting_dataset = meeting_dataset.map(UniformScalingSampler(max_weight=5), )
            if reverb:
                rir_dataset = SMSWSJRIRDatabase(
                    scenarios_json=rir_json_path).get_dataset('test_eval92')
                meeting_dataset = meeting_dataset.map(partial(sample_rirs_for_test_set, rir_dataset=rir_dataset, rng=rng))
            meeting_dataset = meeting_dataset.map(meeting_sampler(test_dataset))


            for meeting in meeting_dataset:
                example_id = meeting['example_id']
                database_dict['datasets'][scenario][example_id] = meeting

        database_dict['meta'][scenario] = overlap_sampler
    pb.io.dump(database_dict, json_path)
    _log.info(f'Wrote file: {json_path}')





