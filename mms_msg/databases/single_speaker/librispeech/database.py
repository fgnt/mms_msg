from lazy_dataset.database import JsonDatabase
from mms_msg import keys
from mms_msg.simulation.utils import load_audio
from paderbox.io.data_dir import database_jsons


class LibriSpeech(JsonDatabase):
    def __init__(self, json_path=database_jsons / 'librispeech.json'):
        super().__init__(json_path)

    def load_example(self, example):
        return load_audio(example, keys.OBSERVATION)


class LibriSpeech8kHz(LibriSpeech):
    def __init__(self, json_path=database_jsons / 'librispeech_8k.json'):
        super().__init__(json_path)
