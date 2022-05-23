import functools
import operator
from dataclasses import dataclass
from typing import Iterable

import lazy_dataset
import paderbox as pb
from mms_msg.sampling.utils import cache_and_normalize_input_dataset, collate_fn
from mms_msg.sampling.utils.rng import get_rng
from lazy_dataset import Dataset
import numpy as np
import logging

logger = logging.getLogger('composition')


def sample_utterance_composition(input_dataset, rng, num_speakers):
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
        input_dataset, rng, num_speakers, *, reduced_set, repetitions: int = 1
):
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
    for _ in range(num_speakers):
        speaker_composition = extend_composition_example_greedy(
            rng, speaker_ids, example_compositions=speaker_composition,
        )

    # Select one random example for each speaker_identifier in each
    # composition
    speaker_composition_example_id = []
    for idx, composition in enumerate(speaker_composition):
        speaker_ids = [speaker_identifier[c] for c in composition]
        speaker_composition_example_id.append([
            grouped_dataset[spk].random_choice(rng_state=rng)['example_id']
            for spk in speaker_ids
        ])

    logger.debug(f'Generated {len(speaker_composition)} speaker '
                 f'compositions')

    return speaker_composition_example_id


@dataclass
class DynamicDataset(Dataset):
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


def _composition_list_to_dict(
        composition: list,
        input_dataset: Dataset,
        dataset_name: str,
) -> dict:
    base = {}
    for idx, composition in enumerate(composition):
        # Combine the sampled examples to one multi-speaker example with a
        # format similar to SMS-WSJ
        example = collate_fn([input_dataset[x] for x in composition])
        example['num_speakers'] = len(example['speaker_id'])
        example['source_dataset'] = example['dataset']

        # The new dataset name is a combination of the given dataset name and
        # the dataset name of the base example. This only works if all examples
        # come from the same source dataset
        assert pb.utils.misc.all_equal(example['source_dataset']), (
            'Dataset name is not equal! Implement something new.'
        )
        example['dataset'] = dataset_name

        # Move audio_path.observation and num_samples to 'original_source' to
        # match SMS-WSJ and to make room for additional keys in the audio_path
        # and num_samples sub-dicts
        example['audio_path'] = {
            'original_source': example['audio_path'].pop('observation')
        }
        example['num_samples'] = {
            'original_source': pb.utils.nested.get_by_path(
                example, 'num_samples.observation'
            )
        }

        # Check that there are no duplicate speakers
        assert pb.utils.misc.all_unique(example['speaker_id']), example['speaker_id']

        # Build an example ID for each example
        example_id = '_'.join([str(idx), *map(str, example['example_id'])])
        assert example_id not in base, (
            'Duplicate example IDs found! Modify the example ID generation '
            'code to avoid this!'
        )
        example['source_id'] = example['example_id']
        example['example_id'] = example_id

        base[example_id] = example
    return base


def get_composition(
        input_dataset: Iterable,
        num_speakers: int,
        composition_sampler=sample_utterance_composition,
        rng: [int, bool] = False,
):
    """
    Build a composition as a `dict` from examples in `input_dataset`.
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
    `input_dataset`
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
        example_compositions = np.arange(len(speaker_ids), dtype=np.int)
        example_compositions = rng.permutation(example_compositions)[:, None]
        return example_compositions

    assert example_compositions.ndim == 2, example_compositions.shape

    given_speaker_ids = [
        set([speaker_ids[c_] for c_ in c]) for c in example_compositions
    ]

    candidates = np.arange(len(speaker_ids), dtype=np.int)
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