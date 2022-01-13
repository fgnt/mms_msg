import numpy as np
from ..utils.sampling import sample_random_round_robin as _sample_random_round_robin
from ..utils.sampling import sample_round_robin as _sample_round_robin
from ..utils.sampling import sample_random as _sample_random


def sample_random(scenarios, examples, rng: np.random.Generator):
    return _sample_random(scenarios, None, rng)


def sample_round_robin(scenarios, examples, rng: np.random.Generator):
    """
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
    return _sample_round_robin(scenarios, examples, rng)


def sample_random_round_robin(scenarios, examples, rng: np.random.Generator):
    """
    >>> scenarios = list('abc')
    >>> examples = []
    >>> rng = np.random.default_rng(0)
    >>> for i in range(10):
    ...     scenario = sample_random_round_robin(scenarios, examples, rng)
    ...     examples.append({'scenario': scenario})
    ...     print(scenario)
    b
    a
    c
    a
    c
    b
    c
    a
    b
    c
    """
    return _sample_random_round_robin(
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


def sample_balanced(scenarios, examples, rng: np.random.Generator):
    activities = _get_activity(scenarios, examples)
    p_scenario = 1 / activities
    p_scenario = p_scenario / np.sum(p_scenario)
    assert np.isclose(np.sum(p_scenario), 1)
    return rng.choice(scenarios, p=p_scenario)


def sample_asymmetric(scenarios, examples, rng: np.random.Generator, target_activity: np.ndarray):
    activities = _get_activity(scenarios, examples)
    activity_diff = target_activity - (
            activities / np.sum(activities)
    )
    p_scenario = np.where(activity_diff > 0, activity_diff, 0)
    p_scenario = p_scenario / np.sum(p_scenario)
    assert np.isclose(np.sum(p_scenario), 1)

    return rng.choice(scenarios, p=p_scenario)
