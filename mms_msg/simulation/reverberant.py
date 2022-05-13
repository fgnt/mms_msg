import numpy as np
from scipy.signal import fftconvolve

from mms_msg import keys
from mms_msg.simulation.anechoic import pad_sparse, get_scale


def reverberant_scenario_map_fn(
        example,
        *,
        normalize_sources: bool = True,
        add_speech_reverberation_early=True,
        add_speech_reverberation_tail=True,
        compensate_time_of_flight=True,
        early_rir_samples: int = int(8000 * 0.05),  # 50 milli seconds
        details=False,
        channel_slice=None,
):
    """
    Modified copy of the scenario_map_fn from sms_wsj.

    This will care for convolution with RIR and also generate noise.
    The random noise generator is fixed based on example ID. It will
    therefore generate the same SNR and same noise sequence the next time
    you use this DB.

    Args:
        compute_scale_on_padded_signals:
        num_channels:
        details:
        early_rir_samples:
        normalize_sources:
        example: Example dictionary.
        sync_speech_source: pad and/or cut the source signal to match the
            length of the observations. Considers the offset.
        add_speech_reverberation_early:
        add_speech_reverberation_tail:
            Calculate the speech_reverberation_tail signal.

    Returns:
        Dict with the following structure:
        ```python
        {
            'audio_data': {
                'observation': array,
                'speech_source': [SparseArray, ...],
                'original_source': [array, ...],
                'speech_image': [SparseArray, ...],
                'noise_image': array,

                # If add_original_reverberated=True
                'original_reverberated': [array, ...],

                # If add_speech_reverberation_early==True
                'speech_reverberation_early: [SparseArray, ...],
                'original_reverberation_early: [SparseArray, ...],

                # If add_speech_reverberation_tail==True
                'speech_reverberation_tail: [SparseArray, ...],
                'original_reverberation_tail': [SparseArray, ...],

                # If add_reverberation_direct==True
                'original_reverberation_direct': [array, ...],
                'speech_reverberation_direct': [SparseArray, ...],
            }
        }
        ```
    """
    audio_data = example[keys.AUDIO_DATA]
    h = audio_data[keys.RIR]  # Shape (K, D, T)
    h = np.asarray(h)

    # Estimate start sample first, to make it independent of channel_mode
    rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in h])

    if channel_slice is not None:
        if isinstance(h, list):
            # All RIRs should have the same length
            h = np.stack(h)
        h = h[:, channel_slice, :]

    _, D, rir_length = h.shape

    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    assert isinstance(early_rir_samples, int), (type(early_rir_samples), early_rir_samples)
    rir_stop_sample = rir_start_sample + early_rir_samples

    # Compute the shifted offsets that align the convolved signals with the
    # speech source
    # This is Jahn's heuristic to be able to still use WSJ alignments.
    if compensate_time_of_flight:
        rir_offset = [
            offset_ - rir_start_sample_
            for offset_, rir_start_sample_ in zip(
                example[keys.OFFSET][keys.ORIGINAL_SOURCE], rir_start_sample)
        ]
    # Don't adapt offsets by subtracting the minimal time of flight (e.g. for multiple microphone arrays)
    else:
        rir_offset = example[keys.OFFSET][keys.ORIGINAL_SOURCE]

    # The two sources have to be cut to same length
    K = len(example[keys.SPEAKER_ID])
    T = example[keys.NUM_SAMPLES][keys.OBSERVATION]
    s = audio_data[keys.ORIGINAL_SOURCE]

    # In some databases (e.g., WSJ) the utterances are not mean normalized. This
    # leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        s = [s_ - np.mean(s_) for s_ in s]

    def get_convolved_signals(h):
        """Convolve the scaled signals `s` with the RIRs in `h`. Returns
        the (unpadded) convolved signals with offsets and the padded convolved
        signals"""
        assert len(s) == len(h), (len(s), len(h))
        x = [
            fftconvolve(s_[..., None, :], h_, axes=-1)
            for s_, h_ in zip(s, h)
        ]

        assert len(x) == len(example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE])
        for x_, T_ in zip(x, example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE]):
            assert x_.shape == (D, T_ + rir_length - 1), (
                x_.shape, D, T_ + rir_length - 1)

        assert len(x) == len(rir_offset) == K
        return x

    # Speech source is simply the shifted and padded original source signals
    audio_data[keys.SPEECH_SOURCE] = pad_sparse(
        audio_data[keys.ORIGINAL_SOURCE],
        example[keys.OFFSET][keys.ORIGINAL_SOURCE],
        target_shape=(T,),
    )

    # Compute the reverberated signals
    audio_data[keys.ORIGINAL_REVERBERATED] = get_convolved_signals(h)

    # Scale s with log_weights before convolution
    scale = get_scale(
        example[keys.LOG_WEIGHTS],
        audio_data[keys.ORIGINAL_REVERBERATED]
    )

    def apply_scale(x):
        return [x_ * scale_ for x_, scale_ in zip(x, scale)]

    audio_data[keys.ORIGINAL_REVERBERATED] = apply_scale(
        audio_data[keys.ORIGINAL_REVERBERATED]
    )

    audio_data[keys.SPEECH_IMAGE] = pad_sparse(
        audio_data[keys.ORIGINAL_REVERBERATED], rir_offset, (D, T))
    example[keys.NUM_SAMPLES][keys.ORIGINAL_REVERBERATED] = [
        a.shape[-1] for a in audio_data[keys.ORIGINAL_REVERBERATED]
    ]
    example[keys.OFFSET][keys.ORIGINAL_REVERBERATED] = rir_offset

    if add_speech_reverberation_early:
        # Obtain the early reverberation part: Mask the tail reverberation by
        # setting everything behind the RIR stop sample to zero
        h_early = h.copy()
        for i in range(h_early.shape[0]):
            h_early[i, ..., rir_stop_sample[i]:] = 0

        # Compute convolution
        audio_data[keys.ORIGINAL_REVERBERATION_EARLY] = apply_scale(
            get_convolved_signals(h_early)
        )
        audio_data[keys.SPEECH_REVERBERATION_EARLY] = pad_sparse(
            audio_data[keys.ORIGINAL_REVERBERATION_EARLY], rir_offset, (D, T))

        if details:
            audio_data[keys.RIR_EARLY] = h_early

    if add_speech_reverberation_tail:
        # Obtain the tail reverberation part: Mask the early reverberation by
        # setting everything before the RIR stop sample to zero
        h_tail = h.copy()
        for i in range(h_tail.shape[0]):
            h_tail[i, ..., :rir_stop_sample[i]] = 0

        # Compute convolution
        audio_data[keys.ORIGINAL_REVERBERATION_TAIL] = apply_scale(
            get_convolved_signals(h_tail)
        )
        audio_data[keys.SPEECH_REVERBERATION_TAIL] = pad_sparse(
            audio_data[keys.ORIGINAL_REVERBERATION_TAIL], rir_offset, (D, T))

        if details:
            audio_data[keys.RIR_TAIL] = h_tail

    clean_mix = sum(audio_data[keys.SPEECH_IMAGE], np.zeros((D, T), dtype=s[0].dtype))
    audio_data[keys.OBSERVATION] = clean_mix
    return example


def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min(
            [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        )

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample
