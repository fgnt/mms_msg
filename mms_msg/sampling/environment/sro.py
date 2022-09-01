import numpy as np
from mms_msg.sampling.utils.rng import get_rng
import paderbox as pb


def sample_sro_weights(example, num_nodes, sto_range, sro_range):
    """
    Samples the average sapling rate offset and sampling time offset for an example
    """
    rng = get_rng(example['example_id'])
    avg_sro = rng.uniform(*sro_range, size=num_nodes)
    sto = rng.randint(*sto_range, size=num_nodes)
    example['avg_sro'] = avg_sro
    example['sto'] = sto
    return example


def sample_sro(example, sigma=.05, theta=.001, max_sro=400):
    """
    Samples a time-varying sampling rate offset (SRO) given an average SRO.
    Sampling is done via an Ornstein-Uhlenbeck process. For more details, see
    https://github.com/gfnt/paderwasn
    """
    from paderwasn.synchronization.simulation import ornstein_uhlenbeck
    sig_len = example['num_samples']['observation']
    sro_seq_len = int(np.ceil((1 + (max_sro * 1e-6)) * sig_len / 2048 + 1))
    avg_sro = example['avg_sro']
    sros = []
    for i, sro in enumerate(avg_sro):
        seed = pb.utils.random_utils.str_to_seed(example['example_id'] + f'_mic_{i}')
        np.random.seed(seed)
        sros.append(ornstein_uhlenbeck(sro_seq_len, sro, sro, sigma, theta))
    example['sro_trajectory'] = sros
    return example


class AsyncParamSampler:
    def __init__(self, num_nodes, sro_range, sto_range, sigma=.05, theta=.001, max_sro=400):
        self.num_nodes = num_nodes
        self.sro_range = sro_range
        self.sto_range = sto_range
        self.sigma = sigma
        self.theta = theta
        self.max_sro = max_sro

    def __call__(self, example):
        example = sample_sro_weights(example, num_nodes=self.num_nodes,
                                     sto_range=self.sto_range, sro_range=self.sro_range)
        example = sample_sro(example, sigma=self.sigma, theta=self.theta, max_sro=self.max_sro)
        return example



