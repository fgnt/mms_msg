import lazy_dataset
import paderbox as pb
from collections import Counter
import numpy as np
import mms_msg


def _generate_input_dataset():
    speakers = list('abcdefghijklmnop')
    rng = np.random.default_rng(42)
    def _gen_example(i):
        num_samples = int(rng.integers(4000, 16000))
        return {
            'num_samples': num_samples,
            'vad': pb.array.interval.ones(num_samples),
            'speaker_id': rng.choice(speakers),
            'example_id': f'id_{i}',
            'audio_path': [],
            'dataset': 'my_dataset',
        }

    return lazy_dataset.new([_gen_example(i) for i in range(30)])


def test_initial_examples():
    # Settings
    num_speakers_in_meeting = 2

    # Generate random input examples
    input_dataset = _generate_input_dataset()

    # Build generator dataset
    ds = mms_msg.get_composition_dataset(input_dataset, num_speakers_in_meeting)
    ds = ds.map(mms_msg.MeetingSampler(duration=30*8000)(input_dataset))

    # Check that distribution of speakers is the same as input dataset
    assert len(ds) == len(input_dataset)
    for i in range(num_speakers_in_meeting):
        assert Counter(ds.map(lambda x: x['speaker_id'][i])) == Counter(input_dataset.map(lambda x: x['speaker_id']))
        assert Counter(ds.map(lambda x: x['speaker_id'][i])) == Counter(input_dataset.map(lambda x: x['speaker_id']))

    # Check that all speakers appear once before the random generation starts
    for ex in ds:
        assert len(set(ex['speaker_id'])) == num_speakers_in_meeting
        assert len(set(ex['speaker_id'][:num_speakers_in_meeting])) == num_speakers_in_meeting


def test_deterministic():
    # Generate random input examples
    input_dataset = _generate_input_dataset()

    # Build generator dataset
    ds1 = mms_msg.get_composition_dataset(input_dataset, 2)
    ds1 = ds1.map(mms_msg.MeetingSampler(duration=30 * 8000)(input_dataset))

    ds2 = mms_msg.get_composition_dataset(input_dataset, 2)
    ds2 = ds2.map(mms_msg.MeetingSampler(duration=30 * 8000)(input_dataset))

    assert list(ds1) == list(ds2)
