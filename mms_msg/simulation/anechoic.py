from typing import List

import numpy as np

from mms_msg import keys
from paderbox.array.sparse import SparseArray


def pad_sparse(original_source, offset, target_shape):
    assert len(offset) == len(original_source), (offset, original_source)
    padded = [
        SparseArray.from_array_and_onset(x_, offset_, target_shape)
        for x_, offset_ in zip(original_source, offset)
    ]

    assert len(padded) == len(original_source), (len(padded), len(original_source))
    for p in padded:
        assert p.shape == target_shape, (p.shape, target_shape)

    return padded


def anechoic_scenario_map_fn(
        example: dict,
        *,
        normalize_sources: bool = True,
) -> dict:
    """
    Constructs the observation and scaled speech source signals for `example`
    for the single-channel no reverberation case.

    Args:
        example: Example dict to load
        normalize_sources: If `True`, the source signals are mean-normalized
            before processing

    Returns:
        Dict with the following structure:
        ```python
        {
            'audio_data': {
                'observation': ndarray,
                'speech_image': [ndarray, ndarray, ...],
                'speech_source': [ndarray, ndarray, ...],
            }
        }
        ```
        where
         - 'observation': The simulated signal observed at a microphone
         - 'speech_source': The shifted source signals
         - 'speech_image': The shifted and scaled source signals. This is
            typically used as a training target signal
    """
    T = example[keys.NUM_SAMPLES][keys.OBSERVATION]
    original_source = example[keys.AUDIO_DATA][keys.ORIGINAL_SOURCE]
    offset = example[keys.OFFSET][keys.ORIGINAL_SOURCE]

    # In some databases (e.g., WSJ) the utterances are not mean normalized. This
    # leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        original_source = [s_ - np.mean(s_) for s_ in original_source]

    # Scale the sources by log_weights. We have to determine the scale based on
    # the full signal (its standard deviation) and not just the cut out part
    scale = get_scale(example[keys.LOG_WEIGHTS], original_source)
    scaled_source = [s_ * scale_ for s_, scale_ in zip(original_source, scale)]

    # Move and pad speech source to the correct position, use sparse array
    speech_source = pad_sparse(original_source, offset, target_shape=(T,))
    speech_image = pad_sparse(scaled_source, offset, target_shape=(T,))

    # The mix is now simply the sum over the speech sources
    mix = sum(speech_source, np.zeros(T, dtype=speech_image[0].dtype))

    example[keys.AUDIO_DATA][keys.OBSERVATION] = mix
    example[keys.AUDIO_DATA][keys.SPEECH_SOURCE] = speech_source
    example[keys.AUDIO_DATA][keys.SPEECH_IMAGE] = speech_image

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

    # Note: scale depends on channel mode
    std = np.maximum(
        np.array([np.std(s, keepdims=True) for s in signals]),
        np.finfo(signals[0].dtype).tiny
    )

    # Bring into the correct shape
    log_weights = log_weights.reshape((-1,) + (1,) * signals[0].ndim)
    scale = 10 ** (log_weights / 20) / std

    # divide by 71 to ensure that all values are between -1 and 1 (WHY 71?)
    # TODO: find a good value for both WSJ and LibriSpeech
    scale /= 71

    return scale
