from pathlib import Path

from lazy_dataset.database import Database
import paderbox as pb


class SMSWSJRIRDatabase(Database):
    @property
    def data(self):
        database_path = Path(self.scenarios_json).parent
        data = pb.io.load(self.scenarios_json)

        def _add_audio_path(example, dataset_name):
            if 'audio_path' not in example:
                example['audio_path'] = {}
            # Scanning the file system is slow
            # example['audio_path']['rir'] = sorted(
            #     (
            #             database_path / dataset_name / example['example_id']
            #     ).glob('*.wav')
            # )
            example['audio_path']['rir'] = [
                database_path / dataset_name / example['example_id'] / f'h_{idx}.wav'
                for idx in range(len(example['source_position'][0]))
            ]
            return example


        data['datasets'] = {
            dataset_name: {
                example_id: _add_audio_path(example, dataset_name)
                for example_id, example in dataset.items()
            } for dataset_name, dataset in data['datasets'].items()
        }

        return data

    def __init__(self, scenarios_json):
        super().__init__()
        self.scenarios_json = scenarios_json
