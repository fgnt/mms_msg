from typing import List, Tuple

import numpy as np
import paderbox as pb
from scipy.signal import fftconvolve
from sms_wsj.database.utils import get_white_noise_for_signal
from sms_wsj.reverb.reverb_utils import get_rir_start_sample

from .. import keys

from paderbox.array.sparse import SparseArray


def synchronize_speech_source_sparse(original_source, offset, T):
    return [
        SparseArray.from_array_and_onset(x_, offset_, T)
        for x_, offset_ in zip(original_source, offset)
    ]


def anechoic_scenario_map_fn(
        example: dict,
        *,
        snr_range: tuple = (20, 30),
        normalize_sources: bool = True,
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

    # Scale the sources by log_weights. We have to determine the scale based on
    # the full signal (its standard deviation) and not just the cut out part
    scale = get_scale(example[keys.LOG_WEIGHTS], s)
    s = [s_ * scale_ for s_, scale_ in zip(s, scale)]

    # Move and pad speech source to the correct position, use sparse array
    x = [
        SparseArray.from_array_and_onset(s_, offset_, T)
        for s_, offset_ in zip(s, offset)
    ]

    # The mix is now simply the sum over the speech sources
    # mix = np.sum(x, axis=0)
    mix = sum(x, np.zeros(T, dtype=s[0].dtype))

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

    Note:
        We assume that the input signals have roughly the same scaling. We
        ignore the scaling of the input signals for computing the log_weights to
        eliminate estimation errors.

        For reference, these are the means of the standard deviations of the
        WSJ database at 8khz:
         - `cv_dev93`: 0.004037793712821765
         - `cv_dev93_5k`: 0.003991357421377814
         - `test_eval92`: 0.016388209061080895
         - `test_eval92_5k`: 0.01724772268374945
         - `test_eval93`: 0.00272367188875606
         - `test_eval93_5k`: 0.0028981842313541535
         - `train_si284`: 0.0061338699176127125
         - `train_si84`: 0.014455413654260894

    Args:
        log_weights: Target logarithmic weights
        signals: The signals to scale. They are used for normalization and to
            obtain the correct shape for the scales.

    Returns:
        Scale, in the same dimensions as `signals`, but as a numpy array.
    """
    assert len(log_weights) == len(signals), (len(log_weights), len(signals))
    log_weights = np.asarray(log_weights)

    # Bring into the correct shape
    log_weights = log_weights.reshape((-1,) + (1,) * signals[0].ndim)
    scale = 10 ** (log_weights / 20)

    # divide by 71 to ensure that all values are between -1 and 1 (WHY 71?)
    # TODO: find a good value for both WSJ and LibriSpeech
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
        # TODO: Handle cut signals, segment offset
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
        normalize_sources: bool = False,
        sync_speech_source=True,
        add_speech_reverberation_early=True,
        add_speech_reverberation_tail=True,
        early_rir_samples: int = int(8000 * 0.05),  # 50 milli seconds
        details=False,
        channel_slice=None,
        squeeze_channels=True,
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

    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    assert isinstance(early_rir_samples, int), (type(early_rir_samples), early_rir_samples)
    rir_stop_sample = rir_start_sample + early_rir_samples

    # The two sources have to be cut to same length
    K = len(example[keys.SPEAKER_ID])
    T = example[keys.NUM_SAMPLES][keys.OBSERVATION]
    s = example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE]

    # In some databases (e.g., WSJ) the utterances are not mean normalized. This
    # leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        s = [s_ - np.mean(s_) for s_ in s]

    # Scale s with log_weights before convolution
    scale = get_scale(example[keys.LOG_WEIGHTS], s)
    s = [s_ * scale_ for s_, scale_ in zip(s, scale)]

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
        x = [
            SparseArray.from_array_and_onset(x_, offset_, (D, T))
            for x_, offset_ in zip(x, offset)
        ]
        assert len(x) == K, (len(x), K)
        assert x[0].shape == (D, T), (x[0].shape, (D, T))
        return x

    x = get_convolved_signals(h)

    example[keys.AUDIO_DATA][keys.SPEECH_IMAGE] = x

    def _squeeze(x):
        if squeeze_channels:
            if channel_slice == slice(1):
                assert x[0].shape[0] == 1
                x = [x_[0] for x_ in x]
        return x

    x = _squeeze(x)

    if add_speech_reverberation_early:
        h_early = h.copy()
        # Replace this with advanced indexing
        for i in range(h_early.shape[0]):
            h_early[i, ..., rir_stop_sample[i]:] = 0
        x_early = get_convolved_signals(h_early)
        x_early = _squeeze(x_early)
        example[keys.AUDIO_DATA][keys.SPEECH_REVERBERATION_EARLY] = x_early

        if details:
            example[keys.AUDIO_DATA][keys.RIR_EARLY] = h_early

    if add_speech_reverberation_tail:
        h_tail = h.copy()
        for i in range(h_tail.shape[0]):
            h_tail[i, ..., :rir_stop_sample[i]] = 0
        x_tail = get_convolved_signals(h_tail)
        x_tail = _squeeze(x_tail)
        example[keys.AUDIO_DATA][keys.SPEECH_REVERBERATION_TAIL] = x_tail

        if details:
            example[keys.AUDIO_DATA][keys.RIR_TAIL] = h_tail

    if sync_speech_source:
        example[keys.AUDIO_DATA][keys.SPEECH_SOURCE] = synchronize_speech_source_sparse(
            example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE],
            offset=example[keys.OFFSET],
            T=T,
        )

    clean_mix = sum(x, np.zeros((D, T), dtype=s[0].dtype))
    example[keys.AUDIO_DATA][keys.OBSERVATION] = clean_mix
    add_microphone_noise(example, snr_range)
    return example
