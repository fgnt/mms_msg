from collections.abc import Hashable

import numpy as np

import lazy_dataset
import paderbox as pb
from mms_msg import keys


def get_rng(*seed: [str, int]) -> np.random.Generator:
    return pb.utils.random_utils.str_to_random_generator(
        '_'.join(map(str, seed)))


def get_rng_state(*seed: [str, int]) -> np.random.RandomState:
    return pb.utils.random_utils.str_to_random_state(
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


def sparse_array_to_numpy(ex, key='audio_data'):
    def func(leaf):
        if 'SparseArray' in leaf.__class__.__name__:
            leaf = np.array(leaf)
        return leaf

    def to_numpy(data):
        return pb.utils.nested.nested_op(func, data)

    audio_data = ex['audio_data']
    for k in audio_data.keys():
        if 'original' in k:
            # original indicate, that the length of the audio depends on the
            # utterance length and not on the mixture length
            #
            # e.g.
            # 'original_source'
            # 'original_reverberated'
            # 'original_reverberation_early'
            # 'original_reverberation_tail'
            # 'original_source'
            audio_data[k] = to_numpy(audio_data[k])
        else:
            # e.g.
            # 'rir'
            # 'noise_image'
            # 'speech_source'
            # 'speech_image'
            # 'speech_reverberation_early'
            # 'speech_reverberation_tail'
            # 'observation'
            audio_data[k] = np.array(to_numpy(audio_data[k]))

    return ex
