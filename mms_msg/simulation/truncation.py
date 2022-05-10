import paderbox as pb
from mms_msg import keys
import numpy as np


def truncate_min(example, *, key='original_source'):
    """
    Truncate a loaded example to the boundaries of the shortest utterance.

    Can be used to create mixtures similar to WSJ0-2mix.
    """
    offset, num_samples = min(
        zip(example['offset'][key], example['num_samples'][key]),
        key=lambda x: x[-1]
    )
    return cut_segment(example, start=offset, stop=offset + num_samples)


def cut_segment(
        source_example: dict, start: int, stop: int,
        clip_offset: bool = False
) -> dict:
    """
    Cut one segment from an example.

    Can be applied before or after loading the example.

    Args:
        source_example: Example to cut a segment from
        start: Start sample of the segment
        stop: Stop sample of the segment (exclusive)
        clip_offset: If true, clip offset at 0 and stop-start. ONLY USE AFTER
            THE EXAMPLE IS LOADED! Otherwise, the positioning information is
            lost for the loading functions

    """
    if stop is None:
        stop = pb.utils.nested.get_by_path(
            source_example, 'num_samples.observation',
            allow_early_stopping=True
        )

    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError(
            f'start and stop must be integers, but got start={start!r} and '
            f'stop={stop!r}'
        )

    # Compute which "utterances" (i.e., entry in original_source) are active in
    # this segment
    # TODO: different keys than original_source?
    active_utterances = [
        offset < stop and start < offset + num_samples
        for offset, num_samples in zip(
            source_example['offset']['original_source'],
            source_example['num_samples']['original_source'],
        )
    ]

    # Shortcut if all entries are empty
    if all([not a for a in active_utterances]):
        return {
            'num_speakers': 0,
            'offset': [],
            'start': start,
            'stop': stop,
        }

    example = pb.utils.nested.nested_op(lambda x: x, source_example)
    flat_example = pb.utils.nested.FlatView(example)

    def apply(fn, keys):
        for key in keys:
            try:
                flat_example[key] = fn(flat_example[key])
            except KeyError:
                # Not all examples contain all keys, ignore missing ones
                pass
            except Exception as e:
                raise RuntimeError(f'Exception in key: {key}') from e

    apply(
        lambda l: [e for e, active in zip(l, active_utterances) if active],
        keys.speaker_wise_keys
    )

    # TODO: generalize the audio keys
    apply(
        lambda a: a[..., start:stop],
        ('audio_data.observation', 'audio_data.noise_image')
    )

    # These have shape (K ... T) where K can be a list
    apply(
        lambda a: np.stack(np.asarray([a_[..., start:stop] for a_ in a])),
        ('audio_data.speech_image', 'audio_data.speech_source')
    )

    # Set some additional info of the example
    example['start'] = start
    example['stop'] = stop
    example['source_example_id'] = source_example['example_id']
    example['num_speakers'] = len(set(example['speaker_id']))

    # Shift offsets
    # TODO: assert clip_offset=True after loading?
    if clip_offset:
        if keys.AUDIO_DATA not in example:
            raise RuntimeError(
                f'Clipping the offsets before loading can change the utterance '
                f'offsets. Either deactivate clipping or '
            )

        def _clip(x):
            if x < 0:
                return 0
            elif x > stop - start:
                return stop - start
            return x
    else:
        def _clip(x):
            return x

    example['offset'] = pb.utils.nested.nested_op(
        lambda o: _clip(o - start), example['offset']
    )
    num_samples = stop - start
    example['num_samples']['observation'] = num_samples

    return example
