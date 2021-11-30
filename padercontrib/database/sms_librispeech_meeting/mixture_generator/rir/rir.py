from dataclasses import dataclass
from pathlib import Path

import lazy_dataset
import paderbox as pb
from lazy_dataset.database import JsonDatabase


def rir_dataset_from_scenarios(
        scenarios_json: [Path, str], dataset_name: str
) -> lazy_dataset.Dataset:
    rir_dataset = JsonDatabase(scenarios_json).get_dataset(dataset_name)

    database_path = Path(scenarios_json).parent

    def _add_audio_path(example):
        if 'audio_path' not in example:
            example['audio_path'] = {}
        # Scanning the file system is slow
        # example['audio_path']['rir'] = sorted(
        #     (
        #             database_path / dataset_name / example['example_id']
        #     ).glob('*.wav')
        # )
        example['audio_path']['rir'] = [
            database_path / dataset_name / example['example_id'] / f'h_{idx}.wav'
            for idx in range(len(example['source_position'][0]))
        ]
        return example

    rir_dataset = rir_dataset.map(_add_audio_path)
    return rir_dataset


def sample_rirs(example: dict, *, rir_dataset: lazy_dataset.Dataset):
    # Assume the examples have a running index
    idx = int(example['example_id'].split('_')[0])
    rir_example = rir_dataset[idx % len(rir_dataset)]
    rirs = rir_example['audio_path']['rir']
    num_speakers = len(example['speaker_id'])
    assert num_speakers <= len(rirs), (num_speakers, len(rirs))
    example['audio_path']['rir'] = rirs[:num_speakers]
    example['room_dimensions'] = rir_example['room_dimensions']
    example['sound_decay_time'] = rir_example['sound_decay_time']
    example['sensor_position'] = rir_example['sensor_position']
    example['source_position'] = [s[:num_speakers] for s in rir_example['source_position']]
    return example


@dataclass(frozen=True)
class RIRSampler:
    rir_dataset: lazy_dataset.Dataset

    @classmethod
    def from_scenarios_json(cls, scenarios_json, dataset_name):
        return cls(rir_dataset_from_scenarios(scenarios_json, dataset_name))

    def __call__(self, example: dict) -> dict:
        return sample_rirs(example, rir_dataset=self.rir_dataset)
