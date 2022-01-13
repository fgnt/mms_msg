from collections import Hashable

import numpy as np

import lazy_dataset
import paderbox as pb
from padercontrib.database import keys


def get_rng(*seed: [str, int]) -> np.random.Generator:
    return pb.utils.random_utils.str_to_random_generator(
        '_'.join(map(str, seed)))


def get_rng_example(example, *seed) -> np.random.Generator:
    return get_rng(example['dataset'], example['example_id'], *seed)


def map_to_spk_id(example, sequence):
    return dict(zip(example['speaker_id'], sequence))


def apply_for_spk(example, sequence):
    sequence = map_to_spk_id(example, sequence)
    return [sequence[s] for s in example['speaker_id']]


def update_num_samples(example):
    example[keys.NUM_SAMPLES][keys.OBSERVATION] = np.max(np.asarray(
        example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE]
    ) + np.asarray(example[keys.OFFSET][keys.ORIGINAL_SOURCE]))


def normalize_example(example):
    """
    The output of this function always has the following structure (plus any
    additional keys that are already present in `example`):

        {
            audio_path: {observation: str},
            num_samples: {observation: int},
            scenario: str,  (only required for meetings, but it doesn't hurt)
            example_id: str,
            speaker_id: str,
            dataset: str,   (source dataset name, required for rng)
        }

    >>> normalize_example({'num_samples': 42, 'audio_path': 'asdf'})
    Traceback (most recent call last):
      ...
    AssertionError: Invalid input example: {'num_samples': 42, 'audio_path': 'asdf'}
    >>> normalize_example({'num_samples': 42, 'audio_path': 'asdf', 'example_id': 0, 'speaker_id': 42})
    {'num_samples': {'observation': 42}, 'audio_path': {'observation': 'asdf'}, 'example_id': 0, 'speaker_id': 42, 'scenario': 42}
    """
    # Perform some checks
    try:
        assert 'example_id' in example
        assert isinstance(example['example_id'], Hashable)
        assert 'speaker_id' in example
        assert isinstance(example['speaker_id'], Hashable)
        assert 'num_samples' in example
        assert 'audio_path' in example
        assert 'dataset' in example
    except AssertionError as e:
        raise AssertionError(f'Invalid input example: {example}') from e

    # Normalize some common formats
    # Introduce the observation key when it is not present (is omitted, e.g.,
    # in LibriSpeech, Timit, ...
    for key in (keys.NUM_SAMPLES, keys.AUDIO_PATH):
        if not isinstance(example[key], dict):
            example[key] = {keys.OBSERVATION: example[key]}

    # Set a default for scenario
    if 'scenario' not in example:
        example['scenario'] = example[keys.SPEAKER_ID]

    return example


def cache_and_normalize_input_dataset(ds):
    if isinstance(ds, dict):
        # Use key as example_id, but overwrite with explicit example_id entry
        ds = [{'example_id': k, **x} for k, x in ds.items()]
    # Cache & sort for reproducibility
    ds = lazy_dataset.from_dict({
        ex['example_id']: ex for ex in ds
    })
    ds = ds.sort().map(normalize_example)
    return ds


def collate_fn(batch):
    """Moves list inside of dict/dataclass recursively.

    Can be used as map after batching of an dataset:
        `dataset.batch(...).map(collate_fn)`

    Args:
        batch: list of examples

    Returns:

    >>> batch = [{'a': 1}, {'a': 2}]
    >>> collate_fn(batch)
    {'a': [1, 2]}
    >>> collate_fn(tuple(batch))
    {'a': (1, 2)}

    >>> batch = [{'a': {'b': [1, 2]}}, {'a': {'b': [3, 4]}}]
    >>> collate_fn(batch)
    {'a': {'b': [[1, 2], [3, 4]]}}

    >>> import dataclasses
    >>> Point = dataclasses.make_dataclass('Point', ['x', 'y'])
    >>> batch = [Point(1, 2), Point(3, 4)]
    >>> batch
    [Point(x=1, y=2), Point(x=3, y=4)]
    >>> collate_fn(batch)
    Point(x=[1, 3], y=[2, 4])
    >>> collate_fn(tuple(batch))
    Point(x=(1, 3), y=(2, 4))
    """
    assert isinstance(batch, (tuple, list)), (type(batch), batch)

    if isinstance(batch[0], dict):
        if len(batch) > 1:
            for b in batch[1:]:
                assert batch[0].keys() == b.keys(), batch
        return batch[0].__class__({
            k: (collate_fn(batch.__class__([b[k] for b in batch])))
            for k in batch[0]
        })
    elif hasattr(batch[0], '__dataclass_fields__'):
        for b in batch[1:]:
            assert batch[0].__dataclass_fields__ == b.__dataclass_fields__, batch
        return batch[0].__class__(**{
            k: (collate_fn(batch.__class__([getattr(b, k) for b in batch])))
            for k in batch[0].__dataclass_fields__
        })
    else:
        return batch


def extend_composition_example_greedy(
        rng, speaker_ids, example_compositions=None, tries=500
):
    """

    Args:
        rng:
        speaker_ids: Speaker id corresponding to an index
        example_compositions:
        tries:

    Returns:

    >>> rng = np.random.RandomState(0)
    >>> speaker_ids = np.array(['Alice', 'Bob', 'Carol', 'Dave', 'Eve'])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids)
    >>> comp
    array([[2],
           [0],
           [1],
           [3],
           [4]])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids, comp)
    >>> comp
    array([[2, 3],
           [0, 4],
           [1, 2],
           [3, 0],
           [4, 1]])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids, comp)
    >>> comp
    array([[2, 3, 1],
           [0, 4, 2],
           [1, 2, 3],
           [3, 0, 4],
           [4, 1, 0]])
    >>> speaker_ids[comp]
    array([['Carol', 'Dave', 'Bob'],
           ['Alice', 'Eve', 'Carol'],
           ['Bob', 'Carol', 'Dave'],
           ['Dave', 'Alice', 'Eve'],
           ['Eve', 'Bob', 'Alice']], dtype='<U5')
    """
    if example_compositions is None:
        example_compositions = np.arange(len(speaker_ids), dtype=np.int)
        example_compositions = rng.permutation(example_compositions)[:, None]
        return example_compositions

    assert example_compositions.ndim == 2, example_compositions.shape

    candidates = np.arange(len(speaker_ids), dtype=np.int)
    speaker_ids = np.array(speaker_ids)
    for _ in range(tries):
        candidates = rng.permutation(candidates)

        try:
            for i in range(len(candidates)):
                for _ in range(tries):
                    if any([
                        speaker_ids[entry_a] == speaker_ids[candidates[i]]
                        for entry_a in example_compositions[i]
                    ]):
                        candidates[i:] = rng.permutation(candidates[i:])
                    else:
                        break

            for tmp in example_compositions.T:
                test_example_composition(tmp, candidates, speaker_ids)

        except AssertionError:
            pass
        else:
            break
    else:
        raise RuntimeError(f'Couldn\'t find a valid speaker composition')

    return np.concatenate([example_compositions, candidates[:, None]], axis=-1)


def test_example_composition(a, b, speaker_ids):
    """

    Args:
        a: List of permutation example indices
        b: List of permutation example indices
        speaker_ids: Speaker id corresponding to an index

    Returns:

    >>> speaker_ids = np.array(['Alice', 'Bob', 'Carol', 'Carol'])
    >>> test_example_composition([0, 1, 2, 3], [2, 3, 1, 0], speaker_ids)
    >>> test_example_composition([0, 1, 2, 3], [1, 0, 3, 2], speaker_ids)
    Traceback (most recent call last):
    ...
    AssertionError: ('speaker duplicate', 2)
    >>> test_example_composition([0, 1, 2, 3], [2, 3, 0, 1], speaker_ids)
    Traceback (most recent call last):
    ...
    AssertionError: ('duplicate pair', 2)



    """
    # Ensure that a speaker is not mixed with itself
    # This also ensures that an utterance is not mixed with itself
    assert np.all(speaker_ids[a] != speaker_ids[b]), (
    'speaker duplicate', len(a) - np.sum(speaker_ids[a] != speaker_ids[b]))

    # Ensure that any pair of utterances does not appear more than once
    tmp = [tuple(sorted(ab)) for ab in zip(a, b)]
    assert len(set(tuple(tmp))) == len(a), ('duplicate pair', len(a) - len(set(tuple(tmp))))
