from lazy_dataset.database import JsonDatabase
from mms_msg import keys
from mms_msg.databases.database import MMSMSGDatabase
from mms_msg.sampling.environment.rir import RIRSampler
from mms_msg.sampling.source_composition import get_composition_dataset
from mms_msg.simulation.anechoic import anechoic_scenario_map_fn
from mms_msg.simulation.noise import white_microphone_noise
from mms_msg.simulation.reverberant import reverberant_scenario_map_fn
from mms_msg.simulation.utils import load_audio
from paderbox.io.data_dir import database_jsons


class AnechoicMeetingDatabase(MMSMSGDatabase):
    def __init__(
            self,
            source_database,
            num_speakers,
            meeting_sampler,
            scaling_sampler,
            snr_sampler,
            source_filter=None,
    ):
        super().__init__(source_database)

        self.num_speakers = num_speakers
        self.meeting_sampler = meeting_sampler
        self.scaling_sampler = scaling_sampler
        self.snr_sampler = snr_sampler
        if source_filter is None:
            def source_filter(_):
                return True
        self.source_filter = source_filter

    def get_mixture_dataset(self, name, rng):
        input_ds = self.source_database.get_dataset(name).filter(self.source_filter)

        ds = get_composition_dataset(
            input_dataset=input_ds,
            num_speakers=self.num_speakers,
            rng=rng
        )
        ds = ds.map(self.scaling_sampler)
        ds = ds.map(self.meeting_sampler(input_ds))
        ds = ds.map(self.snr_sampler)
        return ds

    def load_example(self, example):
        example = load_audio(example, keys.ORIGINAL_SOURCE)
        example = anechoic_scenario_map_fn(example)
        example = white_microphone_noise(example)
        return example


class ReverberantMeetingDatabase(AnechoicMeetingDatabase):

    def __init__(
            self,
            source_database,
            num_speakers,
            meeting_sampler,
            scaling_sampler,
            snr_sampler,
            rir_database,
            source_filter=None
    ):
        super().__init__(
            source_database,
            num_speakers,
            meeting_sampler,
            scaling_sampler,
            snr_sampler,
            source_filter
        )
        self.rir_database = rir_database

    def get_mixture_dataset(self, name, rng):
        input_ds = self.source_database.get_dataset(name).filter(self.source_filter)

        ds = get_composition_dataset(
            input_dataset=input_ds,
            num_speakers=self.num_speakers,
            rng=rng
        )
        ds = ds.map(self.scaling_sampler)
        ds = ds.map(RIRSampler(self.rir_database.get_dataset(name)))
        ds = ds.map(self.meeting_sampler(input_ds))
        ds = ds.map(self.snr_sampler)
        return ds

    def load_example(self, example):
        example = load_audio(example, keys.ORIGINAL_SOURCE, keys.RIR)
        example = reverberant_scenario_map_fn(example)
        example = white_microphone_noise(example)
        return example
