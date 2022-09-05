import numpy as np


def create_async_data(example, n_ch_node=4):
    """
    Desynchronizes the microphone channels of an observation.
    Args:
        example: Meeting example after scenario_map_fn
        n_ch_node: Number of channels per microphone array (necessary for multi-array application)
    Returns: example dict with desynchronized signals as additional key

    """
    from paderwasn.synchronization.simulation import sim_sro
    sros = example['sro_trajectory']
    stos = example['sto']
    async_sigs = []
    sigs = example['audio_data']['observation']
    for i, sig in enumerate(sigs):
        sro = sros[i//n_ch_node]
        sto = stos[i//n_ch_node]
        async_sigs.append(sim_sro(sig[sto:], sro))
    min_len = np.min([len(sig) for sig in async_sigs])
    example['audio_data']['asynchronous_observation'] = \
        np.concatenate([sig[None, :min_len] for sig in async_sigs], axis=0)
    return example

