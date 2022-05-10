import itertools
import operator

import paderbox as pb
from mms_msg import keys


def combine_speaker_signals(example):
    # Group everything by speaker
    grouped_example = {}
    extra = {}

    # Whitelists
    combine_equal = (
        'speaker_id',
        'gender',
    )
    combine_list = (
        'transcription',
        'kaldi_transcription',
    )
    combine = combine_equal + combine_list
    remove = (
        'log_weights',
        'audio_path.rir',
        'source_position',
        'source_dataset',
    )
    blacklist = (
        'sensor_positions',
        'room_dimensions',
        'example_id',
        'num_speakers',
        'sound_decay_time',
        'sensor_position',
    )
    speaker_ids = example[keys.SPEAKER_ID]
    key_fn = operator.itemgetter(0)
    for k, v in pb.utils.nested.flatten(example).items():
        if k in combine:
            if k in combine_equal:
                def _combine(x):
                    assert pb.utils.misc.all_equal(x), x
                    return x[0]
            elif k in combine_list:
                _combine = list
            else:
                assert False, 'Can never happen'

            assert len(speaker_ids) == len(v), (k, len(speaker_ids), len(v))
            grouped_example[k] = [
                _combine(values)
                for _, values in itertools.groupby(sorted(zip(speaker_ids, v), key=key_fn), key=key_fn)
            ]
        elif k in remove:
            # Don't add to grouped example
            pass
        elif k in blacklist:
            extra[k] = v
        else:
            raise RuntimeError(f'key {k} neither in whitelist nor blacklist')