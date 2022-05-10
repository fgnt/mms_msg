import numpy as np
import paderbox as pb


def get_boundaries_from_vad(vad, vad_shift=80, vad_window=200):
    """
    Args:
        vad: Dictionary containing the VAD information for all speech sources
    """
    def apply(example):
        """
        Creates new key 'aligned_source' that describes the source signals without leading and tailing silence.
        """
        example['offset']['aligned_source'] = []
        example['num_samples']['aligned_source'] = []
        for i, example_id in enumerate(example['source_id']):
            try:
                activity = vad[example_id]
                activity = pb.array.interval.ArrayInterval(activity)
                begin = activity.intervals[0][0]
                end = activity.intervals[-1][-1]
                begin = pb.transform.module_stft.stft_frame_index_to_sample_index(begin, vad_window, vad_shift)
                end = pb.transform.module_stft.stft_frame_index_to_sample_index(end, vad_window, vad_shift)
            except:
                print('Found no VAD, using, boundaries')
                begin = 0
                end = example['num_samples']['original_source'][i]

            example['offset']['aligned_source'].append(example['offset']['original_source'][i] + begin)
            example['num_samples']['aligned_source'].append(end-begin)
        return example

    return apply


def get_activity_from_signal(example, signal_name='original_source'):
    """
    Returns the activity intervals for each speaker in an example. The activity intervals are determined as boundaries,
     i.e, only the start and stop values are set, silence regions inside
    the signals are neglected.
    Args:
        example:
        signal_name: Signal, based on which the start time and example lengths are used (e.g. original_reverberated, original_source,
        aligned_source)
    Returns: example dictionary with added key for speaker activity

    """
    speaker_end = [off+num_samples for off, num_samples in zip(example['offset'][signal_name], example['num_samples'][signal_name])]
    array_intervals = {
        spk: pb.array.interval.zeros(int(example['num_samples']['observation'])) for spk in set(example['speaker_id'])
    }

    for spk, start, end in zip(example['speaker_id'], example['offset'][signal_name], speaker_end):
        array_intervals[spk].add_intervals([slice(start, end)])

    example['activity'] = array_intervals
    return example


def get_activity_from_vad(vad, vad_shift, vad_window, signal_name='original_source'):
    """
    Returns the activity intervals for each speaker in an example.
    The activity intervals are determined based on an external vad, and are calculated per frame,
     i.e, silence intervals inside a signal are correctly depicted by the activity
    Args:
        vad: Dictionary containing the vad information for each source signal, can either be saved on a frame-level
             (i.e. frequency-domain) or on a sample level (i.e. time-domain)
        vad_shift: frame shift used for the VAD, if it is determined on a frame level, 1 for sample level
        vad_size: window size of the VAD, if it is determined on a frame level, 1 for sample level
        signal_name: Signal, based on which the VAD was determined
    Returns: example dictionary with added key for speaker activity

    """
    def get_activity(example):
        array_intervals = {
            spk: pb.array.interval.zeros(int(example['num_samples']['observation'])) for spk in
            set(example['speaker_id'])
        }

        for spk, source_id, start in zip(example['speaker_id'], example['source_id'], example['offset'][signal_name]):
            cur_vad = vad[source_id]
            cur_vad = pb.array.interval.ArrayInterval(cur_vad)
            sample_intervals = tuple([tuple(
                [pb.transform.module_stft.stft_frame_index_to_sample_index(boundary, vad_window, vad_shift) + start
                 for boundary in interval]
            ) for interval in cur_vad.intervals])
            array_intervals[spk].add_intervals([slice(i[0], i[1]) for i in sample_intervals])
        example['activity'] = array_intervals

        return example
    return get_activity


def get_silence_intervals(example):
    silence_intervals = pb.array.interval.ones(int(example['num_samples']['observation']))
    for spk in example['activity'].keys():
        silence_intervals.add_intervals([slice(interval[0], interval[1])
                                         for interval in example['activity'][spk].intervals])
    silence_intervals = np.array(silence_intervals)
    silence_intervals = pb.array.interval.ArrayInterval(silence_intervals)
    example['silence'] = silence_intervals
    example['silence_len'] = [interval[1]- interval[0] for interval in silence_intervals.intervals]

    return example


def num_active_spk(example):
    total_activity = np.zeros(example['num_samples']['observation'])
    for spk, activity in example['activity'].items():
        cur_len = len(activity)
        total_activity[:cur_len] = total_activity[:cur_len] + np.array(activity)
    example['num_active_spk'] = total_activity
    return example


def segment_lengths(total_activity, num_spk=0):
    activity_segments = (total_activity == num_spk)
    seg_len = pb.array.interval.ArrayInterval(activity_segments)
    return [seg[1]-seg[0] for seg in seg_len.intervals]


def calculate_overlap(example):
    overlap = example['num_active_spk'] >= 2
    example['overlap_percentage'] = np.sum(overlap) / example['num_samples']['observation']
    overlap = pb.array.interval.ArrayInterval(overlap)
    example['overlap_intervals'] = overlap.intervals
    example['overlap_len'] = [interval[1]-interval[0]
                              for interval in overlap.intervals]
    return example