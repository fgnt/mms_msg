import itertools

import pytest

import mms_msg
import numpy as np


def test_constant_log_weight_sampler():
    # Scalar
    sampler = mms_msg.sampling.environment.scaling.ConstantLogWeightSampler(0.)
    example = {'speaker_id': 'abc'}
    log_weights = sampler(example)['log_weights']
    assert len(log_weights) == len(example['speaker_id'])
    np.testing.assert_equal(np.asarray(log_weights), 0.)

    # Multiple values
    sampler = mms_msg.sampling.environment.scaling.ConstantLogWeightSampler([1., 2., 3., 4., 5.])
    example = {'speaker_id': 'abc'}
    log_weights = sampler(example)['log_weights']
    assert len(log_weights) == len(example['speaker_id'])
    np.testing.assert_equal(np.asarray(log_weights), np.array([-1, 0, 1]))


@pytest.mark.parametrize(
    'num_spk, sampler',
    tuple(itertools.product(
        range(2, 10),
        (mms_msg.sampling.environment.scaling.ConstantLogWeightSampler(),
         mms_msg.sampling.environment.scaling.UniformLogWeightSampler()))
    )
)
def test_symmetrical(num_spk, sampler):
    weights = np.asarray(sampler({'example_id': 0, 'dataset': 0, 'speaker_id': 'a' * num_spk})['log_weights'])
    np.testing.assert_allclose(np.mean(weights), 0, atol=1e-10)


def test_uniform_log_weight_sampler():
    sampler = mms_msg.sampling.environment.scaling.UniformLogWeightSampler()

    # Reproducible and burn test
    np.testing.assert_allclose(
        sampler({'example_id': 0, 'dataset': 0, 'speaker_id': 'a' * 2})['log_weights'],
        [-0.857385, 0.857385], rtol=1e-5)
    np.testing.assert_allclose(
        sampler({'example_id': 0, 'dataset': 0, 'speaker_id': 'a' * 5})['log_weights'],
        [0.033024, 1.747793, -2.318883, -0.425064, 0.96313], rtol=1e-5)


@pytest.mark.parametrize(
    'n_spk, max_difference',
    tuple(itertools.product(
        range(2, 10), range(2, 10)
    ))
)
def test_uniform_log_weight_sampler_in_range(n_spk, max_difference):
    sampler = mms_msg.sampling.environment.scaling.UniformLogWeightSampler(max_weight=max_difference)

    # Test in range
    weights = np.asarray(sampler({
        'example_id': np.random.randint(10000), 'dataset': 0,
        'speaker_id': 'a' * n_spk
    })['log_weights'])
    assert np.max(weights) - np.min(weights) < max_difference
