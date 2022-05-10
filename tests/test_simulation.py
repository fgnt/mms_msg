import copy

import numpy as np
import pytest

import mms_msg

@pytest.fixture
def anechoic_example():
    rng = np.random.default_rng()
    return {
        'example_id': 'test',
        'audio_data': {
            'original_source': [
                rng.random(size=8000),
                rng.random(size=16000),
            ]
        },
        'num_samples': {
            'original_source': [8000, 16000],
            'observation': 17000,
        },
        'log_weights': [0.1, 0.5],
        'offset': {
            'original_source': [2000, 0],
        },
        'snr': 10,
        'dataset': 'test',
    }


@pytest.fixture
def multichannel_example():
    rng = np.random.default_rng()
    return {
        'example_id': 'test',
        'audio_data': {
            'original_source': [
                rng.random(size=8000),
                rng.random(size=16000),
            ],
            'rir': rng.random(size=(2, 6, 8000))
        },
        'num_samples': {
            'original_source': [8000, 16000],
            'observation': 17000,
        },
        'log_weights': [0.1, 0.5],
        'offset': {
            'original_source': [2000, 0],
        },
        'speaker_id': ['a', 'b'],
        'snr': 10,
        'dataset': 'test',
    }


def test_sums_anechoic(anechoic_example):
    original_source = copy.deepcopy(anechoic_example['audio_data']['original_source'])
    out = mms_msg.simulation.anechoic.anechoic_scenario_map_fn(
        anechoic_example, normalize_sources=True
    )
    out = mms_msg.simulation.noise.white_microphone_noise(out)

    audio = out['audio_data']

    # Check num_samples
    np.testing.assert_equal(
        audio['observation'].shape[0], out['num_samples']['observation']
    )
    for actual, desired in zip(
            audio['original_source'],
            out['num_samples']['original_source']
    ):
        np.testing.assert_equal(actual.shape[-1], desired)

    # Check that the original_source is not altered
    for s1, s2 in zip(audio['original_source'], original_source):
        np.testing.assert_allclose(s1, s2)

    # Check that things that should sum up actually sum up to the correct
    # signal
    np.testing.assert_allclose(
        audio['observation'],
        np.sum(audio['speech_image'], axis=0) + audio['noise_image']
    )

    # Check mean normalization works
    for s in audio['speech_source']:
        np.testing.assert_allclose(np.mean(s), 0, atol=1e-10)

    # Check scaling of noise is correct
    np.testing.assert_allclose(
        10 * np.log10(np.sum(
            np.sum(np.array(audio['speech_image']), axis=0) ** 2
        ) / np.sum(audio['noise_image'] ** 2)),
        out['snr'],
    )


def test_sums_multichannel(multichannel_example):
    original_source = copy.deepcopy(multichannel_example['audio_data']['original_source'])
    out = mms_msg.simulation.reverberant.reverberant_scenario_map_fn(
        multichannel_example, normalize_sources=True
    )
    out = mms_msg.simulation.noise.white_microphone_noise(out)

    audio = out['audio_data']

    # Check num samples
    np.testing.assert_equal(
        audio['observation'].shape[-1], out['num_samples']['observation']
    )
    for actual, desired in zip(
            audio['original_reverberated'],
            out['num_samples']['original_reverberated']
    ):
        np.testing.assert_equal(actual.shape[-1], desired)
    for actual, desired in zip(
            audio['original_source'],
            out['num_samples']['original_source']
    ):
        np.testing.assert_equal(actual.shape[-1], desired)

    # Check that the original_source is not altered
    for s1, s2 in zip(audio['original_source'], original_source):
        np.testing.assert_allclose(s1, s2)

    # The convolutions with the RIRs introduce an offset
    # for s in audio['speech_source']:
    #     np.testing.assert_allclose(np.mean(s), 0, atol=1e-10)

    # Check that things that should sum up actually sum up to the correct
    # signal
    np.testing.assert_allclose(
        audio['observation'],
        np.sum(audio['speech_image'], axis=0) + audio['noise_image']
    )
    np.testing.assert_allclose(
        np.stack(audio['speech_image']),
        np.stack(audio['speech_reverberation_early'])
        + np.stack(audio['speech_reverberation_tail']),
        rtol=1e-6
    )

    # Check scaling of noise is correct
    np.testing.assert_allclose(
        10 * np.log10(np.sum(
            np.sum(np.array(audio['speech_image']), axis=0) ** 2
        ) / np.sum(audio['noise_image'] ** 2)),
        out['snr'],
    )


def test_equal(multichannel_example):
    multichannel_example['audio_data']['rir'] = np.ones((2, 1, 1))
    anechoic_out = mms_msg.simulation.anechoic.anechoic_scenario_map_fn(
        multichannel_example, normalize_sources=True
    )
    anechoic_out = mms_msg.simulation.noise.white_microphone_noise(anechoic_out)
    multichannel_out = mms_msg.simulation.reverberant.reverberant_scenario_map_fn(
        multichannel_example, normalize_sources=True
    )
    multichannel_out = mms_msg.simulation.noise.white_microphone_noise(multichannel_out)

    anechoic_audio = anechoic_out['audio_data']
    multichannel_audio = multichannel_out['audio_data']

    for key in ('observation', 'speech_source', 'noise_image', 'speech_image'):
        np.testing.assert_allclose(anechoic_audio[key], multichannel_audio[key])
