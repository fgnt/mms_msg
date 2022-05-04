import numpy as np

import paderbox as pb


def get_rng(*seed: [str, int]) -> np.random.Generator:
    return pb.utils.random_utils.str_to_random_generator(
        '_'.join(map(str, seed)))


def get_rng_state(*seed: [str, int]) -> np.random.RandomState:
    return pb.utils.random_utils.str_to_random_state(
        '_'.join(map(str, seed)))


def get_rng_example(example, *seed) -> np.random.Generator:
    return get_rng(example['dataset'], example['example_id'], *seed)
