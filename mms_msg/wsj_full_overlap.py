from pathlib import Path

from lazy_dataset.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from mms_msg.composition import get_composition_dataset
from mms_msg.rir import RIRSampler
from mms_msg.utils import SMSWSJOffsetSampler, UniformLogWeightSampler
from mms_msg.utils.wsj import filter_punctuation_pronunciation


class WSJ8_kHz_FullOverlap(JsonDatabase):
    def __init__(
            self,
            json_path: [str, Path] = database_jsons / 'wsj_8k.json',
            num_speakers=2,
            max_log_weight=5,
            rng=False,
    ):
        super().__init__(json_path)
        self.num_speakers = num_speakers
        self.rng = rng
        self.max_log_weight = max_log_weight

    def _get_dataset(self, dataset_name=None):
        if not isinstance(dataset_name, str):
            return super()._get_dataset(dataset_name)
        else:
            if 'test' in dataset_name and self.rng is True:
                raise RuntimeError(
                    f'Test datasets should not be generated dynamically (i.e., '
                    f'rng=True)!'
                )
            input_ds = super()._get_dataset(dataset_name)
            input_ds = input_ds.filter(filter_punctuation_pronunciation)

            ds = get_composition_dataset(
                input_dataset=input_ds,
                num_speakers=self.num_speakers,
                rng=self.rng
            )
            ds = ds.map(UniformLogWeightSampler(max_weight=self.max_log_weight))
            ds = ds.map(SMSWSJOffsetSampler())
            return ds


class SpatializedWSJ8_kHz_FullOverlap(WSJ8_kHz_FullOverlap):
    def __init__(
            self,
            rir_json_path: [str, Path],
            json_path: [str, Path] = database_jsons / 'wsj_8k.json',
            num_speakers=2,
            max_log_weight=5,
            rng=False
    ):
        super().__init__(json_path, num_speakers, max_log_weight, rng)
        self.rir_database = JsonDatabase(rir_json_path)

    def _get_dataset(self, dataset_name=None):
        return super()._get_dataset(dataset_name).map(
            RIRSampler(rir_dataset=self.rir_database.get_dataset(dataset_name))
        )
