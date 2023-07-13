import functools
import operator
from dataclasses import dataclass
from typing import Iterable, Union, Tuple

import lazy_dataset
import paderbox as pb
from mms_msg.sampling.utils import cache_and_normalize_input_dataset, collate_fn
from mms_msg.sampling.utils.rng import get_rng, derive_rng
from lazy_dataset import Dataset
import numpy as np
import logging

logger = logging.getLogger('composition')


def sample_utterance_composition(input_dataset, rng, num_speakers):
    """
    Samples a "default" utterance composition.

    Generates one example for each example in the `input_dataset`, so that
    all examples appear equally often (given `num_speakers` is constant).
    Examples are samples so that no two examples from the same speaker are
    paired.
    """
    speaker_ids = [example['speaker_id'] for example in input_dataset]

    # Generate list of example indices for meeting starts
    composition = None
    for _ in range(num_speakers):
        composition = extend_composition_example_greedy(
            rng, speaker_ids, example_compositions=composition,
        )
    logger.debug(f'Generated {len(composition)} speaker '
                 f'compositions')

    return composition


def sample_reduced_utterance_composition(
        input_dataset, rng, num_speakers,
        *,
        reduced_set: Union[callable, str],
        repetitions: int = 1
):
    """
    Samples a shorter "reduced" utterance composition.

    Similar to `sample_utterance_composition`, but generates `repetitions`
    many examples for each unique key returned by `reduced_set`.

    Args:
        reduced_set: Key function or str key to group by
        repetitions: Number of examples generated for every unique key
            returned by `reduced_set`
    """
    if isinstance(reduced_set, str):
        reduced_set = operator.itemgetter(reduced_set)

    # Group the input dataset and create self.num_repetitions many examples
    # for each group
    grouped_dataset = input_dataset.groupby(reduced_set)

    # Make sure that all examples in one group have the same speaker ID
    for dataset in grouped_dataset.values():
        if len(set(dataset.map(lambda x: x['speaker_id']))) != 1:
            raise RuntimeError('Found a group with more than one speaker!')

    speaker_identifier = sorted(grouped_dataset.keys())
    speaker_identifier = speaker_identifier * repetitions

    # Get the compositions based on the group keys. Specific examples are
    # later selected from the groups.
    speaker_ids = [grouped_dataset[identifier][0]['speaker_id']
                   for identifier in speaker_identifier]
    speaker_composition = None
    rng_ = derive_rng(rng)
    for _ in range(num_speakers):
        speaker_composition = extend_composition_example_greedy(
            rng_, speaker_ids, example_compositions=speaker_composition,
        )

    # Select one random example for each speaker_identifier in each
    # composition
    speaker_composition_example_id = []
    for idx, composition in enumerate(speaker_composition):
        rng_ = derive_rng(rng, idx)
        speaker_ids = [speaker_identifier[c] for c in composition]
        speaker_composition_example_id.append([
            grouped_dataset[spk].random_choice(rng_state=rng_)['example_id']
            for spk in speaker_ids
        ])

    logger.debug(f'Generated {len(speaker_composition)} speaker '
                 f'compositions')

    return speaker_composition_example_id


def sample_low_resource_utterance_composition(
        input_dataset, rng: np.random.Generator, num_speakers, *, length,
        deterministic_grow=True,
):
    """
    Args:
        length: Size of the composition, i.e., number of examples to generate
        deterministic_grow: Ensures that the initial examples don't change
            when increasing `length`
    """
    spk_groups = input_dataset.groupby(operator.itemgetter('speaker_id'))
    available_speakers = sorted(spk_groups.keys())

    # Select speakers per example
    if deterministic_grow:
        rng_ = derive_rng(rng)
    else:
        rng_ = rng

    def _choice(rng, x, size):
        """
        A stable random choice that doesn't change the order of the initial
        elements when size changes
        """
        return rng.permutation(x)[:size]

    speaker_constellations = [
        _choice(rng_, available_speakers, num_speakers)
        for _ in range(length)
    ]

    # Select one utterance per speaker in each example
    if deterministic_grow:
        rng_ = derive_rng(rng)
    else:
        rng_ = rng
    examples = [[
        spk_groups[spk].random_choice(rng_state=rng_)['example_id']
        for spk in speaker_constellation
    ] for speaker_constellation in speaker_constellations]

    return examples


def sample_fast_utterance_composition(
        input_dataset, rng: np.random.Generator, num_speakers, *, length=None,
):
    """
    A fast composition sampler that draws random utterances from speaker groups without
    guaranteeing that utterances or speakers appear equally often.

    Args:
        length: Size of the composition, i.e., number of examples to generate
    """
    # Group by speakers
    spk_groups = input_dataset.groupby(operator.itemgetter('speaker_id'))
    available_speakers = sorted(spk_groups.keys())

    if length is None:
        length = len(input_dataset)

    # Draw speakers for each example
    _speakers = rng.permutation(available_speakers)
    speaker_constellations = []
    for _ in range(length):
        if len(_speakers) < num_speakers:
            _speakers = rng.permutation(available_speakers)

        speaker_constellations.append(_speakers[:num_speakers])
        _speakers = _speakers[num_speakers:]

    # Select one utterance per speaker in each example
    _spk_groups = {k: list(v.map(lambda x: x['example_id'])) for k, v in spk_groups.items()}
    _pspk_groups = {k: rng.permutation(v) for k, v in _spk_groups.items()}
    _spk_indices = {k: 0 for k in spk_groups.keys()}
    examples = []
    for speaker_constellation in speaker_constellations:
        example = []
        for speaker in speaker_constellation:
            if _spk_indices[speaker] >= len(_pspk_groups[speaker]):
                _pspk_groups[speaker] = rng.permutation(_spk_groups[speaker])
                _spk_indices[speaker] = 0
            example.append(_pspk_groups[speaker][_spk_indices[speaker]])
            _spk_indices[speaker] += 1
        examples.append(example)
    return examples


@dataclass
class DynamicDataset(Dataset):
    """
    The dataset class used for dynamic mixing. Generates a new utterance
    composition at the start of every epoch.

    Uses `np.random.randint` to sample different seeds for each epoch.
    """
    composition_sampler: callable
    input_dataset: Iterable
    num_speakers: int

    def get_new_dataset(self):
        return get_composition_dataset(
            input_dataset=self.input_dataset,
            composition_sampler=self.composition_sampler,
            rng=int(np.random.randint(2 ** 32)),
            num_speakers=self.num_speakers,
        )

    def copy(self, freeze: bool = False) -> 'lazy_dataset.Dataset':
        if freeze:
            return self.get_new_dataset()
        else:
            return DynamicDataset(
                self.composition_sampler, self.input_dataset, self.num_speakers,
            )

    def __iter__(self, with_key=False):
        return self.get_new_dataset().__iter__(with_key=with_key)

    def __len__(self):
        return len(self.input_dataset)


def collate_example_list(
        examples: list,
        dataset_name: str = None,
        example_id_prefix=None
):
    # Combine the sampled examples to one multi-speaker example with a
    # format similar to SMS-WSJ
    example = collate_fn(examples)
    example['num_speakers'] = len(example['speaker_id'])
    example['source_dataset'] = example['dataset']

    # The new dataset name is a combination of the given dataset name and
    # the dataset name of the base example. This only works if all examples
    # come from the same source dataset
    assert pb.utils.misc.all_equal(example['source_dataset']), (
        'Dataset name is not equal! Implement something new.'
    )
    if dataset_name is not None:
        example['dataset'] = dataset_name

    # Move audio_path.observation and num_samples to 'original_source' to
    # match SMS-WSJ and to make room for additional keys in the audio_path
    # and num_samples sub-dicts
    example['audio_path']['original_source'] = example['audio_path'].pop('observation')
    if 'audio_data' in example:
        example['audio_data']['original_source'] = example['audio_data'].pop('observation')

    example['num_samples'] = {
        'original_source': pb.utils.nested.get_by_path(
            example, 'num_samples.observation'
        )
    }

    # Check that there are no duplicate speakers
    assert pb.utils.misc.all_unique(example['speaker_id']), example['speaker_id']

    # Build an example ID for each example
    example_id_parts = example['example_id']
    if example_id_prefix is not None:
        example_id_parts = (example_id_prefix,) + tuple(example_id_parts)
    example_id = '_'.join(map(str, example_id_parts))

    example['source_id'] = example['example_id']
    example['example_id'] = example_id

    return example


def _composition_list_to_dict(
        composition: list,
        input_dataset: Dataset,
        dataset_name: str,
) -> dict:
    """
    Helper function that builds examples from indices for the `input_dataset`.
    """
    base = {}
    for idx, composition in enumerate(composition):
        example = collate_example_list(
            [input_dataset[x] for x in composition],
            dataset_name=dataset_name,
            example_id_prefix=idx,
        )

        example_id = example['example_id']
        assert example_id not in base, (
            'Duplicate example IDs found! Modify the example ID generation '
            'code to avoid this!'
        )
        base[example_id] = example
    return base


def get_composition(
        input_dataset: Iterable,
        num_speakers: Union[int, Tuple[int, ...]],
        composition_sampler=sample_utterance_composition,
        rng: Union[int, bool] = False,
) -> dict:
    """
    Build a composition as a `dict` from examples in `input_dataset`.

    Note:
        Use the function `get_composition_dataset` if you use `lazy_dataset`.
        The `get_composition` function does not support dynamic mixing because
        it cannot return dynamix mixtures as a dict.

    Args:
        input_dataset: The dataset to draw examples from
        num_speakers: The number of speakers. It can either be an int or a
            tuple of ints, in which case the number of speakers is randomly
            drawn from that list for each example.
            As an example, `num_speakers=(1, 2, 2, 3)` would generate
            1-spk and 3-spk compositions with a probability of 0.25 each and
            2-spk compositions with a probability of 0.5.
        composition_sampler: The composition sampling algorithm.
            See `sample_utterance_composition`,
            `sample_reduced_utterance_composition`
            and `sample_low_resource_utterance_composition`
        rng: Either an `int` or `False`. If it is an int, it is used as a
            seed for generating the composition. `True` is only supported by
            `get_composition_dataset`.
    """
    input_dataset = cache_and_normalize_input_dataset(input_dataset)

    # Infer name from dataset. Make sure that all examples come
    # from the same dataset (otherwise the dataset name is not unique)
    name = input_dataset[0]['dataset']
    assert all([x['dataset'] == name for x in input_dataset])

    # Construct a name and rng from the `rng` parameter
    if rng is True:
        raise ValueError(
            f'rng=True only works with get_composition_dataset, '
            f'not with get_composition'
        )
    elif rng is False:
        pass
    elif isinstance(rng, int):
        name += f'_rng{rng}'
    else:
        raise TypeError(rng)
    rng = get_rng('composition', name)

    # Sample the composition
    if isinstance(num_speakers, int):
        max_speakers = num_speakers
    else:
        max_speakers = max(num_speakers)

    composition = composition_sampler(
        input_dataset=input_dataset, rng=rng,
        num_speakers=max_speakers
    )

    # Sample the number of speakers
    if not isinstance(num_speakers, int):
        new_composition = []
        for c in composition:
            num_speakers_ = rng.choice(num_speakers)
            new_composition.append(c[:num_speakers_])
        composition = new_composition

    # Convert the composition to the correct format
    composition = _composition_list_to_dict(
        composition, input_dataset, name
    )
    return composition


def get_composition_dataset(
        input_dataset: Iterable,
        num_speakers: int,
        composition_sampler: callable = sample_utterance_composition,
        rng: [bool, int] = False,
):
    """
    Build a composition as a `lazy_dataset.Dataset` from examples in
    `input_dataset`.

    Note:
        If you don't want to use `lazy_dataset`, use `get_composition`.
        Be aware that `get_composition` doesn't support dynamic mixing.

    Args:
        input_dataset: The dataset to draw examples from
        num_speakers: The number of speakers. It can either be an int or a
            tuple of ints, in which case the number of speakers is randomly
            drawn from that list for each example.
            As an example, `num_speakers=(1, 2, 2, 3)` would generate
            1-spk and 3-spk compositions with a probability of 0.25 each and
            2-spk compositions with a probability of 0.5.
        composition_sampler: The composition sampling algorithm.
            See `sample_utterance_composition`,
            `sample_reduced_utterance_composition`
            and `sample_low_resource_utterance_composition`
        rng: Either an `int` or `False`. If it is an int, it is used as a
            seed for generating the composition. `True` is only supported by
            `get_composition_dataset`.
    """
    if rng is True:
        return DynamicDataset(
            composition_sampler,
            cache_and_normalize_input_dataset(input_dataset),
            num_speakers,
        )
    composition = get_composition(
        input_dataset=input_dataset,
        num_speakers=num_speakers,
        composition_sampler=composition_sampler,
        rng=rng,
    )
    name = next(iter(composition.values()))['dataset']
    return lazy_dataset.new(composition, name=name)


def get_reduced_composition_dataset(
        input_dataset: lazy_dataset.Dataset,
        num_speakers: int,
        reduced_set: [str, callable],
        repetitions: int = 1,
        rng: [int, bool] = False,
):
    return get_composition_dataset(
        input_dataset, num_speakers,
        functools.partial(
            sample_reduced_utterance_composition, reduced_set=reduced_set,
            repetitions=repetitions
        ),
        rng
    )


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
        example_compositions = np.arange(len(speaker_ids), dtype=int)
        example_compositions = rng.permutation(example_compositions)[:, None]
        return example_compositions

    assert example_compositions.ndim == 2, example_compositions.shape

    given_speaker_ids = [
        set([speaker_ids[c_] for c_ in c]) for c in example_compositions
    ]

    candidates = np.arange(len(speaker_ids), dtype=int)
    speaker_ids = np.array(speaker_ids)
    for _ in range(tries):
        candidates = rng.permutation(candidates)

        try:
            for i in range(len(candidates)):
                for _ in range(tries):
                    if speaker_ids[candidates[i]] in given_speaker_ids[i]:
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
