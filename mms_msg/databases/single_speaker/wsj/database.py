from lazy_dataset.database import JsonDatabase
from mms_msg import keys
from mms_msg.simulation.utils import load_audio
from paderbox.io.data_dir import database_jsons


class WSJ(JsonDatabase):
    def __init__(self, json_path=database_jsons / 'wsj.json'):
        super().__init__(json_path)

    def load_example(self, example):
        return load_audio(example, keys.OBSERVATION)


class WSJ8kHz(WSJ):
    def __init__(self, json_path=database_jsons / 'wsj_8k.json'):
        super().__init__(json_path)
