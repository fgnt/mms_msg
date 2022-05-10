import numpy as np

from mms_msg import keys
from mms_msg.sampling.utils.rng import get_rng_example


def get_white_noise_for_signal(
        time_signal: np.ndarray,
        *,
        snr: float,
        rng: np.random.Generator = np.random,
):
    """
    Args:
        time_signal:
        snr: SNR or single speaker SNR.
        rng: A random number generator object
    """
    noise_signal = rng.normal(size=time_signal.shape)

    power_time_signal = np.mean(time_signal ** 2, keepdims=True)
    power_noise_signal = np.mean(noise_signal ** 2, keepdims=True)
    current_snr = 10 * np.log10(power_time_signal / power_noise_signal)

    factor = 10 ** (-(snr - current_snr) / 20)

    noise_signal *= factor
    return noise_signal


def white_microphone_noise(example: dict) -> dict:
    """
    Adds microphone noise to `example`.

    Uses the example ID in `example` for RNG seeding.
    Modifies `example` in place.

    Adds the key "audio_data.noise_image" to the example.

    Args:
        example: The example to add microphone noise to. Must contain the keys
            "snr" and "audio_data.observation"

    """
    if keys.SNR not in example:
        raise KeyError(f'Example must provide an SNR. Use the SNR sampler '
                       f'from `mms_msg.sampling.environment.noise`.')
    if keys.AUDIO_DATA not in example \
            or keys.OBSERVATION not in example[keys.AUDIO_DATA]:
        raise KeyError(f'An observation signal must be present in the example')

    observation = example[keys.AUDIO_DATA][keys.OBSERVATION]
    n = get_white_noise_for_signal(
        observation,
        snr=example['snr'],
        rng=get_rng_example(example),
    )
    example[keys.AUDIO_DATA][keys.NOISE_IMAGE] = n
    example[keys.AUDIO_DATA][keys.OBSERVATION] = observation + n
    return example
