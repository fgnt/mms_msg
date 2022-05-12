from pathlib import Path
from typing import Callable

import numpy as np

import lazy_dataset.database
from lazy_dataset import Dataset
from lazy_dataset.database import JsonDatabase, Database
from mms_msg.databases.database import MMSMSGDatabase
from mms_msg import keys
from mms_msg.sampling.environment.rir import RIRSampler
from mms_msg.sampling.source_composition import get_composition_dataset
from mms_msg.simulation.anechoic import anechoic_scenario_map_fn
from mms_msg.simulation.noise import white_microphone_noise
from mms_msg.simulation.reverberant import reverberant_scenario_map_fn
from mms_msg.simulation.truncation import truncate_min
from mms_msg.simulation.utils import load_audio


class AnechoicSpeakerMixtures(MMSMSGDatabase):
    def __init__(
            self,
            source_database: lazy_dataset.database.Database,
            num_speakers: int,
            offset_sampler: Callable[[dict], dict],
            scaling_sampler: Callable[[dict], dict],
            truncate_to_shortest: bool = True,
            source_filter: Callable[[dict], bool] = None,
    ):
        """
        Base database class for classical anechoic speech mixtures.

        Mixtures generated with this class have the following properties:
         - one utterance per speaker
         - no reverberation
         - no noise

         Args:
             source_json_path: Path to the source database json
             num_speakers: Number of speakers per mixture
             offset_sampler: A sampling module to sample an offset
                (key 'offset.original_source') for each utterance
            scaling_sampler: A sampling module that samples scaling weights
                (key 'log_weights')
            truncate_to_shortest: Inspired by WSJ0-2/3mix. If 'min', the mixture is truncated
                to the shorter utterance to ensure full overlap. If 'max',
                utterances are not truncated
            source_filter: A function to filter the source examples
        """
        super().__init__(source_database)
        self.num_speakers = num_speakers
        self.overlap_sampler = offset_sampler
        self.scaling_sampler = scaling_sampler
        self.truncate_to_shortest = truncate_to_shortest
        if source_filter is None:
            def source_filter(_):
                return True
        self.source_filter = source_filter

    def get_mixture_dataset(self, name: str, rng: np.random.Generator) -> Dataset:
        ds = get_composition_dataset(
            input_dataset=self.source_database.get_dataset(name).filter(self.source_filter),
            num_speakers=self.num_speakers,
            rng=rng
        )
        ds = ds.map(self.scaling_sampler)
        ds = ds.map(self.overlap_sampler)
        return ds

    def load_example(self, example: dict) -> dict:
        example = load_audio(example, keys.ORIGINAL_SOURCE)
        example = anechoic_scenario_map_fn(example)
        if self.truncate_to_shortest:
            example = truncate_min(example)
        return example


class ReverberantSpeakerMixtures(AnechoicSpeakerMixtures):
    def __init__(self,
                 source_database: lazy_dataset.database.Database,
                 num_speakers: int,
                 offset_sampler: Callable[[dict], dict],
                 scaling_sampler: Callable[[dict], dict],
                 rir_database: Database,
                 snr_sampler: Callable[[dict], dict],
                 truncate_to_shortest: bool = True,
                 source_filter: Callable[[dict], bool] = None):
        """
        Base database class for classical reverberant speech mixtures.

        Mixtures generated with this class have the following properties:
         - one utterance per speaker
         - reverberation with room impulse response
         - white microphone noise

          Args:
             source_json_path: Path to the source database json
             num_speakers: Number of speakers per mixture
             offset_sampler: A sampling module to sample an offset
                (key 'offset.original_source') for each utterance
            scaling_sampler: A sampling module that samples scaling weights
                (key 'log_weights')
            rir_database: Database object that contains RIRs
            snr_sampler: A sampling module that samples an SNR for the
                microphone noise
            truncate_to_shortest: Inspired by WSJ0-2/3mix. If 'min', the mixture is truncated
                to the shorter utterance to ensure full overlap. If 'max',
                utterances are not truncated
        """
        super().__init__(source_database, num_speakers, offset_sampler, scaling_sampler, truncate_to_shortest, source_filter)
        self.rir_database = rir_database
        self.snr_sampler = snr_sampler

    def get_mixture_dataset(self, name: str, rng: np.random.Generator) -> Dataset:
        return super().get_mixture_dataset(name, rng).map(
            RIRSampler(self.rir_database.get_dataset(name))
        )

    def load_example(self, example: dict) -> dict:
        example = load_audio(example, keys.ORIGINAL_SOURCE, keys.RIR)
        example = reverberant_scenario_map_fn(example)
        example = white_microphone_noise(example)
        if self.truncate_to_shortest:
            example = truncate_min(example)
        return example
