from pathlib import Path

from lazy_dataset.database import JsonDatabase
from mms_msg.databases.utils import get_dataset_name_and_rng
from paderbox.io.data_dir import database_jsons
from mms_msg.sampling.source_composition import get_composition_dataset
from mms_msg.sampling.environment.rir import RIRSampler
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.databases.single_speaker.wsj.utils import filter_punctuation_pronunciation
from mms_msg.sampling.pattern.meeting import MeetingSampler
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.environment.noise import UniformSNRSampler
from .database import AnechoicMeetingDatabase, ReverberantMeetingDatabase
from paderbox.io import data_dir
from ..reverberation.sms_wsj import SMSWSJRIRDatabase

OVERLAP_SETTINGS = {
    'no_ov': {
        'max_concurrent_spk': 2,
        'p_silence': 1,
        'maximum_silence': 2 * 8000,
        'maximum_overlap': 0,
    },
    'medium_ov': {
        'max_concurrent_spk': 2,
        'p_silence': 0.1,
        'maximum_silence': 2 * 8000,
        'maximum_overlap': 8 * 8000,
    },
    'high_ov': {
        'max_concurrent_spk': 2,
        'p_silence': 0.01,
        'maximum_silence': 1 * 8000,
        'maximum_overlap': 8 * 8000,
        'hard_minimum_overlap': 2 * 8000,
    }
}


def AnechoicWSJ8kHzMeeting(source_json_path=database_jsons / 'wsj_8k.json', duration=120 * 8000,
                           overlap_conditions='medium_ov', num_speakers=(5, 6, 7, 8)):
    """
    Meetings based on the WSJ0-2mix dataset. The resulting mixtures will have a matching value range,
    so that models trained on this data can be evaluated on WSJ0-2mix and vice versa.

    Args:
        num_speaker:
        overlap_conditions:

    Returns:

    """
    if isinstance(overlap_conditions, str):
        try:
            overlap_conditions = OVERLAP_SETTINGS[overlap_conditions]
        except:
            raise KeyError(f'No settings defined for overlap scenario {overlap_conditions}') from None

    overlap_sampler = UniformOverlapSampler(**overlap_conditions)
    meeting_sampler = MeetingSampler(duration, overlap_sampler)
    return AnechoicMeetingDatabase(source_json_path=source_json_path,
                                   num_speakers=num_speakers,
                                   meeting_sampler=meeting_sampler,
                                   scaling_sampler=UniformScalingSampler(5),
                                   snr_sampler=UniformSNRSampler(20, 30))


def ReverberantWSJ8kHzMeeting(source_json_path=database_jsons / 'wsj_8k.json', duration=120 * 8000,
                              overlap_conditions='medium_ov', num_speakers=(5, 6, 7, 8),
                              scenario_json_path=data_dir.db_dir / 'sms_wsj' / 'rirs' / 'scenarios.json'):
    """
    Meetings based on the WSJ0-2mix dataset. The resulting mixtures will have a matching value range,
    so that models trained on this data can be evaluated on WSJ0-2mix and vice versa.

    Args:
        num_speaker:
        overlap_conditions:

    Returns:

    """
    if isinstance(overlap_conditions, str):
        try:
            overlap_conditions = OVERLAP_SETTINGS[overlap_conditions]
        except:
            raise KeyError(f'No settings defined for overlap scenario {overlap_conditions}') from None

    overlap_sampler = UniformOverlapSampler(**overlap_conditions)
    meeting_sampler = MeetingSampler(duration, overlap_sampler)
    return ReverberantMeetingDatabase(source_json_path=source_json_path,
                                      num_speakers=num_speakers,
                                      meeting_sampler=meeting_sampler,
                                      scaling_sampler=UniformScalingSampler(5),
                                      snr_sampler=UniformSNRSampler(20, 30),
                                      rir_database=SMSWSJRIRDatabase(scenario_json_path))


class WSJ_8kHz_Meeting(JsonDatabase):
    """
    >>> db = WSJ_8kHz_Meeting(num_speakers=4)
    >>> from pprint import pprint
    >>> ex = db.get_dataset('cv_dev93')[0]
    >>> ex['dataset'], ex['example_id'], sorted(set(ex['speaker_id']))
    ('cv_dev93', '0_4k4c030k_4k1c030f_4k2c030p_4kac0318', ['4k1', '4k2', '4k4', '4ka'])
    >>> ex = db.get_dataset('cv_dev93_rng42')[0]
    >>> ex['dataset'], ex['example_id'], sorted(set(ex['speaker_id']))
    ('cv_dev93_rng42', '0_4kac030t_4k4c030v_4k3c0318_4k7c031a', ['4k3', '4k4', '4k7', '4ka'])
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> ex = next(iter(db.get_dataset('cv_dev93_rng')))
    >>> ex['dataset'], ex['example_id'], sorted(set(ex['speaker_id']))
    ('cv_dev93_rng2357136044', '0_4k9c031d_4k2c0317_4k3c030p_4k8c0305', ['4k2', '4k3', '4k8', '4k9'])
    """

    def __init__(
            self,
            json_path: [str, Path] = database_jsons / 'wsj_8k.json',
            meeting_sampler=MeetingSampler(),
            num_speakers=(5, 6, 7, 8),
            max_log_weight=5,
            rir_scenarios_json_path=None,
    ):
        super().__init__(json_path)
        self.num_speakers = num_speakers
        self.max_log_weight = max_log_weight
        self.meeting_sampler = meeting_sampler
        self.rir_scenarios_json_path = rir_scenarios_json_path

    def _get_dataset(self, dataset_name=None):
        if not isinstance(dataset_name, str):
            return super()._get_dataset(dataset_name)
        else:
            dataset_name, rng = get_dataset_name_and_rng(dataset_name)
            input_ds = super()._get_dataset(dataset_name)
            input_ds = input_ds.filter(filter_punctuation_pronunciation)

            ds = get_composition_dataset(
                input_dataset=input_ds,
                num_speakers=self.num_speakers,
                rng=rng
            )
            ds = ds.map(UniformScalingSampler(max_weight=self.max_log_weight))
            if self.rir_scenarios_json_path is not None:
                ds = ds.map(RIRSampler.from_scenarios_json(
                    self.rir_scenarios_json_path, dataset_name
                ))
            ds = ds.map(self.meeting_sampler(input_ds))
            return ds
