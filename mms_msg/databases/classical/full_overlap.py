from mms_msg.databases.classical.database import AnechoicSpeakerMixtures, ReverberantSpeakerMixtures
from mms_msg.databases.reverberation.sms_wsj import SMSWSJRIRDatabase
from mms_msg.databases.single_speaker.wsj.utils import filter_punctuation_pronunciation
from mms_msg.sampling.environment.noise import UniformSNRSampler
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.sampling.pattern.classical import ConstantOffsetSampler, SMSWSJOffsetSampler
from paderbox.io import data_dir
from paderbox.io.data_dir import database_jsons


def WSJ2Mix(source_json_path=database_jsons / 'wsj_8k.json', mode='min'):
    """
    A database similar to the WSJ(0)-2mix database
    """
    return AnechoicSpeakerMixtures(
        source_json_path=source_json_path,
        num_speakers=2,
        offset_sampler=ConstantOffsetSampler(0),
        scaling_sampler=UniformScalingSampler(5),
        mode=mode,
        source_filter=filter_punctuation_pronunciation,
    )


def WSJ3Mix(source_json_path=database_jsons / 'wsj_8k.json', mode='min'):
    """
    A database similar to the WSJ(0)-2mix database
    """
    return AnechoicSpeakerMixtures(
        source_json_path=source_json_path,
        num_speakers=3,
        offset_sampler=ConstantOffsetSampler(0),
        scaling_sampler=UniformScalingSampler(5),
        mode=mode,
        source_filter=filter_punctuation_pronunciation,
    )


def Libri2MixClean(source_json_path=database_jsons / 'librispeech.json', mode='min'):
    return AnechoicSpeakerMixtures(
        source_json_path=source_json_path,
        num_speakers=3,
        offset_sampler=ConstantOffsetSampler(0),
        scaling_sampler=UniformScalingSampler(5),
        mode=mode,
    )


def SMSWSJ(
        source_json_path=database_jsons / 'wsj_8k.json',
        scenario_json_path=data_dir.db_dir / 'sms_wsj' / 'rirs' / 'scenarios.json',
        num_speakers=2,
):
    return ReverberantSpeakerMixtures(
        source_json_path=source_json_path,
        num_speakers=num_speakers,
        overlap_sampler=SMSWSJOffsetSampler(),
        scaling_sampler=UniformScalingSampler(5),
        snr_sampler=UniformSNRSampler(20, 30),
        rir_database=SMSWSJRIRDatabase(scenario_json_path),
        source_filter=filter_punctuation_pronunciation,
    )
