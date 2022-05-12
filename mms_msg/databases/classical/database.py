from lazy_dataset.database import JsonDatabase
from mms_msg.databases.database import MMSMSGDatabase
from mms_msg import keys
from mms_msg.sampling.environment.rir import RIRSampler
from mms_msg.sampling.source_composition import get_composition_dataset
from mms_msg.simulation.anechoic import anechoic_scenario_map_fn
from mms_msg.simulation.noise import white_microphone_noise
from mms_msg.simulation.reverberant import reverberant_scenario_map_fn
from mms_msg.simulation.truncation import truncate_min
from mms_msg.simulation.utils import load_audio


class AnechoicSpeakerMixtures(JsonDatabase, MMSMSGDatabase):
    def __init__(
            self,
            source_json_path,
            num_speakers,
            offset_sampler,
            scaling_sampler,
            mode='min',
            source_filter=None,
    ):
        super().__init__(source_json_path)
        self.num_speakers = num_speakers
        self.overlap_sampler = offset_sampler
        self.scaling_sampler = scaling_sampler
        self.mode = mode
        if source_filter is None:
            def source_filter(_):
                return True
        self.source_filter = source_filter

    def get_mixture_dataset(self, name, rng):
        ds = get_composition_dataset(
            input_dataset=super().get_dataset(name).filter(self.source_filter),
            num_speakers=self.num_speakers,
            rng=rng
        )
        ds = ds.map(self.scaling_sampler)
        ds = ds.map(self.overlap_sampler)
        return ds

    def load_example(self, example: dict) -> dict:
        example = load_audio(example, keys.ORIGINAL_SOURCE)
        example = anechoic_scenario_map_fn(example)
        if self.mode == 'min':
            example = truncate_min(example)
        return example


class ReverberantSpeakerMixtures(AnechoicSpeakerMixtures):
    def __init__(self,
                 source_json_path,
                 num_speakers,
                 overlap_sampler,
                 scaling_sampler,
                 rir_database,
                 snr_sampler,
                 mode='min',
                 source_filter=None):
        super().__init__(source_json_path, num_speakers, overlap_sampler, scaling_sampler, mode, source_filter)
        self.rir_database = rir_database
        self.snr_sampler = snr_sampler

    def get_mixture_dataset(self, name, rng):
        return super().get_mixture_dataset(name, rng).map(
            RIRSampler(self.rir_database.get_dataset(name))
        )

    def load_example(self, example: dict) -> dict:
        example = load_audio(example, keys.ORIGINAL_SOURCE, keys.RIR)
        example = reverberant_scenario_map_fn(example)
        example = white_microphone_noise(example)
        if self.mode == 'min':
            example = truncate_min(example)
        return example
