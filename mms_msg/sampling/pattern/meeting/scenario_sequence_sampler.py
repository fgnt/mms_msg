import numpy as np
from mms_msg.sampling.utils import sequence_sampling
from paderbox.utils.mapping import Dispatcher


__all__ = [
    'sample_random',
    'sample_round_robin',
    'sample_random_round_robin',
    'sample_balanced',
    'sample_balanced_no_repeat',
    'sample_asymmetric',
    'scenario_sequence_samplers',
]


def sample_random(scenarios: list, examples: list, rng: np.random.Generator):
    """
    Samples the scenarios uniformly, independent of the already sampled
    `examples`.

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the key `'scenario'`.
        rng: A random number generator
    """
    return sequence_sampling.sample_random(scenarios, None, rng)


def sample_round_robin(scenarios: list, examples: list, rng: np.random.Generator):
    """
    Samples the scenarios in a repeating round-robin pattern.

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the key `'scenario'`. Used to determine the current state in the
            round-robin sequence.
        rng: A random number generator

    >>> scenarios = list('abc')
    >>> for i in range(10):
    ...     print(sample_round_robin(scenarios, [{}]*i, None))
    a
    b
    c
    a
    b
    c
    a
    b
    c
    a
    """
    return sequence_sampling.sample_round_robin(scenarios, examples, rng)


def sample_random_round_robin(scenarios: list, examples: list, rng: np.random.Generator):
    """
    Samples the scenarios in rounds, but each round is sampled uniformly.
    Each round has length `len(scenario)` and each scenario is samples exactly
    once in each round.

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the key `'scenario'`. Used to determine which examples have already
            been sampled in the current round.
        rng: A random number generator

    >>> scenarios = list('abc')
    >>> examples = []
    >>> rng = np.random.default_rng(42)
    >>> for i in range(10):
    ...     scenario = sample_random_round_robin(scenarios, examples, rng)
    ...     examples.append({'scenario': scenario})
    ...     print(scenario)
    a
    c
    b
    b
    a
    c
    b
    c
    a
    a
    """
    return sequence_sampling.sample_random_round_robin(
        scenarios, [x['scenario'] for x in examples], rng
    )


def _get_activity(scenarios, examples):
    return np.asarray([
        sum([
            x['num_samples']['observation']
             for x in examples if x['scenario'] == scenario
        ])
        for scenario in scenarios
    ])


def sample_balanced(scenarios: list, examples: list, rng: np.random.Generator, *, eps=1e-7):
    """
    Samples the scenarios so that their activity (in num samples) is balanced.

    Samples so that the number of samples for each scenario is approximately
    equal (we can only guarantee that when sampling continues forever).

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the keys `'scenario'` and `'num_samples.observation'`. Used to
            find the current activity for each scenario.
        rng: A random number generator
        eps: Used as the activity when a speaker is not active. A small value
            should have no effect. A large value changes the initial sampling.
            This can be set to a large value (e.g., average utterance length)
            when no enrollment phase is used
            and the initial sampling should be random.

    Comment on `eps`:
    The sampler receivers empty activities when the enrollment phase is disabled in the
    meeting sampler, thus the `eps` is required to avoid division by zero.
    It makes sense to set `eps` to a large value (e.g., average utterance length) when
    an enrollment phase is not desired since it is still very likely that
    the initial sampling will be the same as the enrollment phase (i.e., all speakers
    are active once in a row before a more realistic sampling begins) when `eps` is small.
    Setting `eps` to a value that is much larger than the meeting length would result
    in an unbalanced sampling similar to `sample_random`, but when `eps` is set to a
    reasonable value, the initial sampling will be random and the sampling will become
    balanced after a few examples, which is more realistic than the enrollment phase.
    """
    activities = _get_activity(scenarios, examples)
    # Avoid division by zero. This can happen when a speaker has not been active
    # A small epsilon shouldn't change the result
    activities = np.where(activities > 0, activities, eps)
    p_scenario = 1 / activities
    p_scenario = p_scenario / np.sum(p_scenario)
    assert np.isclose(np.sum(p_scenario), 1)
    return rng.choice(scenarios, p=p_scenario)


def sample_balanced_no_repeat(scenarios: list, examples: list, rng: np.random.Generator, *, eps=1e-7):
    """
    Samples the scenarios so that their activity (in num samples) is balanced, but disallow
    repetitions of the same scenario.

    Samples so that the number of samples for each scenario is approximately
    equal (we can only guarantee that when sampling continues forever).

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the keys `'scenario'` and `'num_samples.observation'`. Used to
            find the current activity for each scenario.
        rng: A random number generator
        eps: Used as the activity when a speaker is not active. A small value
            should have no effect. A large value changes the initial sampling.
            This can be set to a large value (e.g., average utterance length)
            when no enrollment phase is used
            and the initial sampling should be random.
    """
    activities = _get_activity(scenarios, examples)
    # Avoid division by zero. This can happen when a speaker has not been active
    # A small epsilon shouldn't change the result
    activities = np.where(activities > 0, activities, eps)
    p_scenario = 1 / activities
    if examples:
        last_scenario = sorted(examples, key=lambda x: x['offset']['original_source'] + x['num_samples']['observation'])[-1]['scenario']
        p_scenario[scenarios.index(last_scenario)] = 0
    p_scenario = p_scenario / np.sum(p_scenario)
    assert np.isclose(np.sum(p_scenario), 1)
    return rng.choice(scenarios, p=p_scenario)


def sample_asymmetric(scenarios, examples, rng: np.random.Generator, target_activity: np.ndarray):
    """
    Samples so that the activity (in num samples) roughly matches `target_activity`.

    Sampling is done similar to `sample_balanced`, but the target is not given
    by `target_activity` instead of a uniform distribution.

    Args:
        scenarios: The scenarios to sample from
        examples: The examples already sampled. Should be a list of dicts with
            the keys `'scenario'` and `'num_samples.observation'`. Used to
            find the current activity for each scenario.
        rng: A random number generator
    """
    activities = _get_activity(scenarios, examples)
    activity_diff = target_activity - (
            activities / np.sum(activities)
    )
    p_scenario = np.where(activity_diff > 0, activity_diff, 0)
    p_scenario = p_scenario / np.sum(p_scenario)
    assert np.isclose(np.sum(p_scenario), 1)

    return rng.choice(scenarios, p=p_scenario)


scenario_sequence_samplers = Dispatcher({
    'random': sample_random,
    'round_robin': sample_round_robin,
    'random_round_robin': sample_random_round_robin,
    'balanced': sample_balanced,
    'balanced_no_repeat': sample_balanced_no_repeat,
})
