from lazy_dataset.database import JsonDatabase
from mms_msg import keys
from mms_msg.simulation.utils import load_audio
from paderbox.io.data_dir import database_jsons


class LibriSpeech(JsonDatabase):
    def __init__(self, json_path=database_jsons / 'librispeech.json', scenario_key=None):
        super().__init__(json_path)
        if isinstance(scenario_key, str):
            scenario_key = (scenario_key,)
        self.scenario_key = scenario_key

    def load_example(self, example):
        return load_audio(example, keys.OBSERVATION)

    def add_scenario(self, example):
        example['scenario'] = '_'.join([example[key] for key in sorted(self.scenario_key)])
        return example

    def get_dataset(self, name=None):
        dataset = super().get_dataset(name)
        if self.scenario_key is not None:
            dataset = dataset.map(self.add_scenario)
        return dataset


class LibriSpeech8kHz(LibriSpeech):
    def __init__(self, json_path=database_jsons / 'librispeech_8k.json', scenario_key=None):
        super().__init__(json_path, scenario_key=scenario_key)
