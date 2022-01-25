import functools

import pytest

import lazy_dataset
import mms_msg
from collections import Counter


def get_input_dataset():
    return lazy_dataset.new([
        {
            'speaker_id': speaker_id,
            'example_id': i,
            'dataset': 'my_dataset',
            'num_samples': 42,
            'audio_path': [],
        } for i, speaker_id in enumerate('abcdefg' * 3)
    ])


def test_full_speaker_composition_sampler():
    input_dataset = get_input_dataset()
    num_speakers = 2
    composition = mms_msg.get_composition(input_dataset, num_speakers)

    # Check length
    assert len(composition) == len(input_dataset)

    # Check that distribution of examples is the same as input dataset
    example_ids = input_dataset.map(lambda x: x['example_id'])
    for i in range(num_speakers):
        assert Counter(example_ids) == Counter([v['source_id'][i] for v in composition.values()])


def test_reduced_speaker_composition_sampler():
    repetitions = 2
    input_dataset = get_input_dataset()
    num_speakers = 2
    composition = mms_msg.get_composition(
        input_dataset, num_speakers,
        composition_sampler=functools.partial(
            mms_msg.composition.sample_reduced_utterance_composition,
            reduced_set='speaker_id', repetitions=2,
        ),
    )

    speaker_ids = input_dataset.map(lambda x: x['speaker_id'])
    # Check length and that distribution of speakers is the same as input dataset
    assert len(composition) == len(set(speaker_ids)) * repetitions
    for i in range(num_speakers):
        assert {k: v * repetitions for k, v in Counter(set(speaker_ids)).items()} == Counter([v['speaker_id'][i] for v in composition.values()])


@pytest.mark.parametrize(
    'sampler_fn', (
        mms_msg.composition.sample_utterance_composition,
        functools.partial(
            mms_msg.composition.sample_reduced_utterance_composition,
            reduced_set='speaker_id'
        )
    )
)
def test_varying_num_speakers(sampler_fn):
    input_dataset = get_input_dataset()

    # Check that the first examples are equal when compositions with the same
    # input dataset but different numers of speakers are generated
    ref_spk = []
    for num_speakers in range(2, 6):
        composition = mms_msg.get_composition(input_dataset, num_speakers, sampler_fn)
        example = next(iter(composition.values()))
        assert example['speaker_id'][:len(ref_spk)] == ref_spk
        ref_spk = example['speaker_id']


def test_deterministic():
    input_dataset = get_input_dataset()

    # Initialize two times with the same seed should give the same examples
    composition1 = mms_msg.get_composition(input_dataset, 2)
    composition2 = mms_msg.get_composition(input_dataset, 2)
    composition3 = mms_msg.get_composition(input_dataset, 2, rng=1234)
    composition4 = mms_msg.get_composition(input_dataset, 2, rng=1234)

    assert composition1 == composition2
    assert composition3 == composition4
    assert composition2 != composition3


def test_dynamic_mixing():
    input_dataset = get_input_dataset()

    # Initialize two times with the same seed should give the same examples
    ds1 = mms_msg.get_composition_dataset(input_dataset, 2, rng=True)

    # Every time ds is iterated it generates a new sequence
    assert list(ds1) != list(ds1)
    assert list(ds1) != list(ds1)
    assert list(ds1) != list(ds1)
