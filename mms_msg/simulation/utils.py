import paderbox as pb
from mms_msg import keys


def load_audio(example, *load_keys):
    audio_path = example[keys.AUDIO_PATH]
    audio_data = example[keys.AUDIO_DATA] = {}
    for k in load_keys:
        audio_data[k] = pb.utils.nested.nested_op(pb.io.load_audio, audio_path[k])
    return example
