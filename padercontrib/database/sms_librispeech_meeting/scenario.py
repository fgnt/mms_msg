from typing import List, Tuple

import numpy as np
import paderbox as pb
from scipy.signal import fftconvolve
from sms_wsj.database.utils import extract_piece, get_white_noise_for_signal, synchronize_speech_source
from sms_wsj.reverb.reverb_utils import get_rir_start_sample

from .. import keys


def anechoic_scenario_map_fn(
        example: dict,
        *,
        snr_range: tuple = (20, 30),
        normalize_sources: bool = True,
        # This should never be set to False, it is just here for
        # backwards-compatibility
        compute_scale_on_padded_signals: bool = False,
) -> dict:
    """
    Constructs the observation and scaled speech source signals for `example`
    for the single-channel no reverberation case.

    Args:
        example: Example dict to load
        snr_range: Range where SNR is sampled from for white microphone noise.
            This is deterministic; the rng is seeded with the example ID
        normalize_sources: If `True`, the source signals are mean-normalized
            before processing
        compute_scale_on_padded_signals: Whether to compute the scale
            normalization for the log_weights on the original source signals
            or on the padded and shifted signals. This option can have a large
            impact on the overall loudness of the resulting audio.
            Only set this to `True` if you know what you are doing!
    """
    T = example[keys.NUM_SAMPLES][keys.OBSERVATION]
    s = example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE]
    offset = example[keys.OFFSET]

    # In some databases (e.g., WSJ) the utterances are not mean normalized. This
    # leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        s = [s_ - np.mean(s_) for s_ in s]

    # Move and pad speech source to the correct position
    x = [extract_piece(s_, offset_, T) for s_, offset_ in zip(s, offset)]
    x = np.stack(x)

    # Scale the sources by log_weights. We have to determine the scale based on
    # the full signal (its standard deviation) and not just the cut out part
    if compute_scale_on_padded_signals:
        scale = get_scale(example[keys.LOG_WEIGHTS], x)
    else:
        scale = get_scale(example[keys.LOG_WEIGHTS], s)
    x *= scale

    # The mix is now simply the sum over the speech sources
    mix = np.sum(x, axis=0)

    example[keys.AUDIO_DATA][keys.OBSERVATION] = mix
    example[keys.AUDIO_DATA][keys.SPEECH_IMAGE] = x

    # Anechoic case: Speech image == speech source
    example[keys.AUDIO_DATA][keys.SPEECH_SOURCE] = x

    # Add noise if snr_range is specified. RNG depends on example ID (
    # deterministic)
    add_microphone_noise(example, snr_range)

    return example


def get_scale(
        log_weights: List[float], signals: List[np.ndarray]
) -> np.ndarray:
    """
    Computes the normalized scales for all signals so that multiplying them with
    the scales gives a logarithmic ratio of log_weights.

    Args:
        log_weights: Target logarithmic weights
        signals: The signals to scale. They are used for normalization and to
            obtain the correct shape for the scales.

    Returns:
        Scale, in the same dimensions as `signals`, but as a numpy array.
    """
    assert len(log_weights) == len(signals), (len(log_weights), len(signals))

    std = np.maximum(np.array(
        [np.std(s, axis=-1, keepdims=True) for s in signals]
    ), np.finfo(signals[0].dtype).tiny)

    log_weights = np.asarray(log_weights)

    # Bring into the correct shape
    log_weights = log_weights.reshape((-1,) + (1,) * signals[0].ndim)

    scale = (10 ** (log_weights / 20)) / std

    # divide by 71 to ensure that all values are between -1 and 1 (WHY 71?)
    scale /= 71

    return scale


def add_microphone_noise(example: dict, snr_range: Tuple[int, int]):
    """
    Adds microphone noise to `example`. Uses the example ID in `example` for
    RNG seeding.

    Modifies `example` in place.

    Args:
        example: The example to add microphone noise to
        snr_range: Range for uniformly drawing SNR. If `None`, no noise is
            added.
    """
    if snr_range is not None:
        example_id = example[keys.EXAMPLE_ID]
        rng = pb.utils.random_utils.str_to_random_generator(example_id)
        example[keys.SNR] = snr = rng.uniform(*snr_range)

        rng = pb.utils.random_utils.str_to_random_generator(example_id)
        mix = example[keys.AUDIO_DATA][keys.OBSERVATION]
        n = get_white_noise_for_signal(mix, snr=snr, rng_state=rng)
        example[keys.AUDIO_DATA][keys.NOISE_IMAGE] = n
        mix += n
        example[keys.AUDIO_DATA][keys.OBSERVATION] = mix


def multi_channel_scenario_map_fn(
        example,
        *,
        snr_range: tuple = (20, 30),
        normalize_sources: bool = True,
        sync_speech_source=True,
        add_speech_reverberation_early=True,
        add_speech_reverberation_tail=True,
        early_rir_samples: int = int(8000 * 0.05),  # 50 milli seconds
        details=False,
        channel_slice=None,
        squeeze_channels=True,
        # This should never be set to False, it is just here for
        # backwards-compatibility
        compute_scale_on_padded_signals: bool = False,
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
        snr_range: required for noise generation
        sync_speech_source: pad and/or cut the source signal to match the
            length of the observations. Considers the offset.
        add_speech_reverberation_early:
        add_speech_reverberation_tail:
            Calculate the speech_reverberation_tail signal.

    Returns:

    """
    h = example[keys.AUDIO_DATA][keys.RIR]  # Shape (K, D, T)

    # Estimate start sample first, to make it independent of channel_mode
    rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in h])

    if channel_slice is not None:
        h = h[:, channel_slice, :]

    _, D, rir_length = h.shape

    # TODO: SAMPLE_RATE not defined
    # rir_stop_sample = rir_start_sample + int(SAMPLE_RATE * 0.05)
    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    assert isinstance(early_rir_samples, int), (type(early_rir_samples), early_rir_samples)
    rir_stop_sample = rir_start_sample + early_rir_samples

    # The two sources have to be cut to same length
    K = len(example[keys.SPEAKER_ID])
    T = example[keys.NUM_SAMPLES][keys.OBSERVATION]
    if keys.ORIGINAL_SOURCE not in example[keys.AUDIO_DATA]:
        # legacy code
        example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE] = example[keys.AUDIO_DATA][keys.SPEECH_SOURCE]
    if keys.ORIGINAL_SOURCE not in example[keys.NUM_SAMPLES]:
        # legacy code
        example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE] = example[keys.NUM_SAMPLES][keys.SPEECH_SOURCE]

    s = example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE]

    # In some databases (e.g., WSJ) the utterances are not mean normalized. This
    # leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        s = [s_ - np.mean(s_) for s_ in s]

    def get_convolved_signals(h):
        assert len(s) == len(h), (len(s), len(h))
        x = [
            fftconvolve(s_[..., None, :], h_, axes=-1)
            for s_, h_ in zip(s, h)
        ]

        assert len(x) == len(example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE])
        for x_, T_ in zip(x, example[keys.NUM_SAMPLES][keys.ORIGINAL_SOURCE]):
            assert x_.shape == (D, T_ + rir_length - 1), (
                x_.shape, D, T_ + rir_length - 1)

        # This is Jahn's heuristic to be able to still use WSJ alignments.
        offset = [
            offset_ - rir_start_sample_
            for offset_, rir_start_sample_ in zip(
                example[keys.OFFSET], rir_start_sample)
        ]

        assert len(x) == len(offset)
        x = [extract_piece(x_, offset_, T) for x_, offset_ in zip(x, offset)]
        x = np.stack(x, axis=0)
        assert x.shape == (K, D, T), (x.shape, (K, D, T))
        # TODO: is this correct?
        return x

    x = get_convolved_signals(h)

    # Scale the sources by log_weights
    if compute_scale_on_padded_signals:
        scale = get_scale(example[keys.LOG_WEIGHTS], x)
    else:
        # s is not convolved yet, so we have to add the channel dimension
        scale = get_scale(example[keys.LOG_WEIGHTS], s)[:, None, :]
    x *= scale

    def _squeeze(signal):
        if squeeze_channels:
            if channel_slice == slice(1):
                assert signal.shape[1] == 1
                signal = signal[:, 0, :]
        return signal

    x = _squeeze(x)

    example[keys.AUDIO_DATA][keys.SPEECH_IMAGE] = x

    if add_speech_reverberation_early:
        h_early = h.copy()
        # Replace this with advanced indexing
        for i in range(h_early.shape[0]):
            h_early[i, ..., rir_stop_sample[i]:] = 0
        x_early = get_convolved_signals(h_early)
        x_early *= scale
        x_early = _squeeze(x_early)
        example[keys.AUDIO_DATA][keys.SPEECH_REVERBERATION_EARLY] = x_early

        if details:
            example[keys.AUDIO_DATA][keys.RIR_EARLY] = h_early

    if add_speech_reverberation_tail:
        h_tail = h.copy()
        for i in range(h_tail.shape[0]):
            h_tail[i, ..., :rir_stop_sample[i]] = 0
        x_tail = get_convolved_signals(h_tail)
        x_tail *= scale
        x_tail = _squeeze(x_tail)
        example[keys.AUDIO_DATA][keys.SPEECH_REVERBERATION_TAIL] = x_tail

        if details:
            example[keys.AUDIO_DATA][keys.RIR_TAIL] = h_tail

    if sync_speech_source:
        example[keys.AUDIO_DATA][keys.SPEECH_SOURCE] = synchronize_speech_source(
            example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE],
            offset=example[keys.OFFSET],
            T=T,
        )
    else:
        # legacy code
        example[keys.AUDIO_DATA][keys.SPEECH_SOURCE] = \
            example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE]

    clean_mix = np.sum(x, axis=0)
    example[keys.AUDIO_DATA][keys.OBSERVATION] = clean_mix
    add_microphone_noise(example, snr_range)
    return example
