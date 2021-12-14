import functools
from pathlib import Path

from paderbox.io.data_dir import database_jsons
from padercontrib.io import data_dir
from padercontrib.database.wsj import WSJ_8kHz
from .composition import get_composition_dataset
from .rir.rir import RIRSampler
from .utils.scaling import UniformLogWeightSampler
from .utils.wsj import filter_punctuation_pronunciation
from .meeting import MeetingSampler


class WSJ8_kHz_Meeting(WSJ_8kHz):
    def __init__(
            self,
            json_path: [str, Path] = database_jsons / 'wsj_8k.json',
            alignment_handler=None,
            meeting_sampler=MeetingSampler(),
            num_speakers=(5, 6, 7, 8),
            max_log_weight=5,
            rng=False,
            rir_scenarios_json_path=None,
    ):
        super().__init__(json_path, alignment_handler)
        self.num_speakers = num_speakers
        self.rng = rng
        self.max_log_weight = max_log_weight
        self.meeting_sampler = meeting_sampler
        self.rir_scenarios_json_path = rir_scenarios_json_path

    @staticmethod
    def format_fn(example):
        example['num_samples'] = example['num_samples']['observation']
        example['scenario'] = example['speaker_id']
        return example

    def _get_dataset(self, dataset_name=None):
        if not isinstance(dataset_name, str):
            return super()._get_dataset(dataset_name)
        else:
            if 'test' in dataset_name and self.rng is True:
                raise RuntimeError(
                    f'Test datasets should not be generated dynamically (i.e., '
                    f'rng=True)!'
                )
            input_ds = super()._get_dataset(dataset_name)
            input_ds = input_ds.filter(filter_punctuation_pronunciation)
            input_ds = input_ds.map(self.format_fn)

            ds = get_composition_dataset(
                input_dataset=input_ds,
                num_speakers=self.num_speakers,
                rng=self.rng
            )
            ds = ds.map(UniformLogWeightSampler(max_weight=self.max_log_weight))
            if self.rir_scenarios_json_path is not None:
                ds = ds.map(RIRSampler.from_scenarios_json(
                    self.rir_scenarios_json_path, dataset_name
                ))
            ds = ds.map(self.meeting_sampler(input_ds))
            return ds
