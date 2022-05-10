from dataclasses import dataclass
from pathlib import Path

import lazy_dataset
from lazy_dataset.database import JsonDatabase
from mms_msg.databases.reverberation.sms_wsj import SMSWSJRIRDatabase


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
        return cls(SMSWSJRIRDatabase(scenarios_json).get_dataset(dataset_name))

    def __call__(self, example: dict) -> dict:
        return sample_rirs(example, rir_dataset=self.rir_dataset)
