from lazy_dataset.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.sampling.pattern.meeting import MeetingSampler
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.environment.noise import UniformSNRSampler
from .database import AnechoicMeetingDatabase, ReverberantMeetingDatabase
from paderbox.io import data_dir
from ..reverberation.sms_wsj import SMSWSJRIRDatabase
from ..single_speaker.wsj.utils import filter_punctuation_pronunciation

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


def AnechoicWSJ8kHzMeeting(source_json_path=database_jsons / 'wsj_8k.json',
                           duration=120 * 8000,
                           overlap_conditions='medium_ov',
                           num_speakers=(5, 6, 7, 8),
                           scenario_sequence_sampler='balanced',
                           ):
    """
    Meetings based on the WSJ0-2mix dataset. The resulting mixtures will have a matching value range,
    so that models trained on this data can be evaluated on WSJ0-2mix and vice versa.

    Args:
        source_json_path: Path to the JSON file created for the WSJ data
        duration: Minimal duration of each meeting (in samples). The sampling of new utterances is stopped once the meeting
                  length exceeds this value
        overlap_conditions: Specifies the overlap scenario, either via pre-defined scnearios or custom values
            either str or dict of overlap settings
        num_speakers: Number of speakers per meeting. Any permitted number of speakers needs to be listed.
        scenario_json_path: Path to the 'scenarios.json' that is created after simulating the SMSWSJ RIRs

    Returns:
        Database object containing configurations for anechoic WSJ meetings
    """
    if isinstance(overlap_conditions, str):
        try:
            overlap_conditions = OVERLAP_SETTINGS[overlap_conditions]
        except:
            raise KeyError(f'No settings defined for overlap scenario {overlap_conditions}') from None

    overlap_sampler = UniformOverlapSampler(**overlap_conditions)
    meeting_sampler = MeetingSampler(
        duration, overlap_sampler=overlap_sampler,
        scenario_sequence_sampler=scenario_sequence_sampler,
    )
    return AnechoicMeetingDatabase(source_database=JsonDatabase(source_json_path),
                                   num_speakers=num_speakers,
                                   meeting_sampler=meeting_sampler,
                                   scaling_sampler=UniformScalingSampler(5),
                                   snr_sampler=UniformSNRSampler(20, 30),
                                   source_filter=filter_punctuation_pronunciation,
                                   )


def ReverberantWSJ8kHzMeeting(source_json_path=database_jsons / 'wsj_8k.json',
                              duration=120 * 8000,
                              overlap_conditions='medium_ov', num_speakers=(5, 6, 7, 8),
                              scenario_sequence_sampler='balanced',
                              scenario_json_path=data_dir.db_dir / 'sms_wsj' / 'rirs' / 'scenarios.json'):
    """
    Meetings based on the WSJ0-2mix dataset. The resulting mixtures will have a matching value range,
    so that models trained on this data can be evaluated on SMS-WSJ and vice versa.

    Args:
        source_json_path: Path to the JSON file created for the WSJ data
        duration: Minimal duration of each meeting (in samples). The sampling of new utterances is stopped once the meeting
                  length exceeds this value
        overlap_conditions: Specifies the overlap scenario, either via pre-defined scnearios or custom values
            either str or dict of overlap settings
        num_speakers: Number of speakers per meeting. Any permitted number of speakers needs to be listed.
        scenario_json_path: Path to the 'scenarios.json' that is created after simulating the SMSWSJ RIRs

    Returns:
        Database object containing configurations for reverberated WSJ meetings
    """
    if isinstance(overlap_conditions, str):
        try:
            overlap_conditions = OVERLAP_SETTINGS[overlap_conditions]
        except:
            raise KeyError(f'No settings defined for overlap scenario {overlap_conditions}') from None

    overlap_sampler = UniformOverlapSampler(**overlap_conditions)
    meeting_sampler = MeetingSampler(
        duration, overlap_sampler=overlap_sampler,
        scenario_sequence_sampler=scenario_sequence_sampler,
    )
    return ReverberantMeetingDatabase(source_database=JsonDatabase(source_json_path),
                                      num_speakers=num_speakers,
                                      meeting_sampler=meeting_sampler,
                                      scaling_sampler=UniformScalingSampler(5),
                                      snr_sampler=UniformSNRSampler(20, 30),
                                      rir_database=SMSWSJRIRDatabase(scenario_json_path),
                                      source_filter=filter_punctuation_pronunciation,
                                      )
