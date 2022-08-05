import numpy as np


def sample_random_round_robin(a, sequence, rng: np.random.Generator):
    """
    >>> a = list('abc')
    >>> sequence = []
    >>> rng = np.random.default_rng(42)
    >>> for i in range(10):
    ...     s = sample_random_round_robin(a, sequence, rng)
    ...     sequence.append(s)
    ...     print(s)
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
    # TODO: what if a contains duplicate elements?
    present_elements = set(sequence[len(sequence)-(len(sequence) % len(a)):])
    return rng.choice(sorted(set(a) - present_elements))


def sample_random(a, sequence, rng: np.random.Generator):
    return rng.choice(a)


def sample_round_robin(a, sequence, rng: np.random.Generator):
    """
    >>> a = list('abc')
    >>> for i in range(10):
    ...     print(sample_round_robin(a, [{}]*i, None))
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
    return a[len(sequence) % len(a)]
