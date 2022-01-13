import functools
from pathlib import Path

from paderbox.io.data_dir import database_jsons
from padercontrib.io import data_dir
from padercontrib.database.wsj import WSJ_8kHz
from padercontrib.database.sms_librispeech_meeting.mixture_generator.composition import get_composition_dataset
from padercontrib.database.sms_librispeech_meeting.mixture_generator.rir.rir import RIRSampler
from padercontrib.database.sms_librispeech_meeting.mixture_generator.utils.scaling import UniformLogWeightSampler
from padercontrib.database.sms_librispeech_meeting.mixture_generator.utils.wsj import filter_punctuation_pronunciation
from padercontrib.database.sms_librispeech_meeting.mixture_generator.meeting import MeetingSampler


def get_dataset_name_and_rng(dataset_name):
    if 'rng' in dataset_name:
        if 'test' in dataset_name:
            raise ValueError(
                f'Dynamic mixing should not be activated on test '
                f'datasets to ensure reproducibility (i.e., no "rng" '
                f'in the dataset name: {dataset_name})'
            )

        try:
            dataset_name, seed = dataset_name.split('_rng')
        except ValueError:
            raise ValueError(
                f'Expected "<original_dataset_name>_rng[seed]" '
                f'(e.g., train_si284_rng), not {dataset_name}'
            ) from None

        if seed != '':
            rng = int(seed)
        else:
            rng = True
    else:
        rng = False
    return dataset_name, rng


class WSJ_8kHz_Meeting(WSJ_8kHz):
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
            alignment_handler=None,
            meeting_sampler=MeetingSampler(),
            num_speakers=(5, 6, 7, 8),
            max_log_weight=5,
            rir_scenarios_json_path=None,
    ):
        super().__init__(json_path, alignment_handler)
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
            ds = ds.map(UniformLogWeightSampler(max_weight=self.max_log_weight))
            if self.rir_scenarios_json_path is not None:
                ds = ds.map(RIRSampler.from_scenarios_json(
                    self.rir_scenarios_json_path, dataset_name
                ))
            ds = ds.map(self.meeting_sampler(input_ds))
            return ds
