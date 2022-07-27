from lazy_dataset.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from mms_msg.sampling.environment.scaling import UniformScalingSampler
from mms_msg.sampling.pattern.meeting import MeetingSampler
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.environment.noise import UniformSNRSampler
from .database import AnechoicMeetingDatabase
from .wsj_meeting import OVERLAP_SETTINGS


def AnechoicLibriSpeechMeeting(source_json_path=database_jsons / 'librispeech.json',
                           duration=120 * 16000, overlap_conditions='medium_ov', num_speakers=(5, 6, 7, 8)):
    """
    Meetings based on the LibriSpeech corpus.

    Args:
        source_json_path: Path to the JSON file created for the WSJ data
        duration: Minimal duration of each meeting (in samples). The sampling of new utterances is stopped once the meeting
                  length exceeds this value
        overlap_conditions: Specifies the overlap scenario, either via pre-defined scenarios or custom values
            either str or dict of overlap settings
        num_speakers: Number of speakers per meeting. Any permitted number of speakers needs to be listed.

    Returns:
        Database object containing configurations for anechoic LibriSpeech meetings
    """
    if isinstance(overlap_conditions, str):
        try:
            overlap_conditions = OVERLAP_SETTINGS[overlap_conditions]
        except:
            raise KeyError(f'No settings defined for overlap scenario {overlap_conditions}') from None

    overlap_sampler = UniformOverlapSampler(**overlap_conditions)
    meeting_sampler = MeetingSampler(duration, overlap_sampler=overlap_sampler)
    return AnechoicMeetingDatabase(source_database=JsonDatabase(source_json_path),
                                   num_speakers=num_speakers,
                                   meeting_sampler=meeting_sampler,
                                   scaling_sampler=UniformScalingSampler(5),
                                   snr_sampler=UniformSNRSampler(20, 30),
                                   )