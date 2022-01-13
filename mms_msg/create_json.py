import copy
import datetime

from sacred import Experiment
from tqdm import tqdm

from mms_msg.databases import WSJ8_kHz_FullOverlap
import paderbox as pb
import padertorch as pt

ex = Experiment('mixture_generator_create_json')


@ex.config
def defaults():
    json_path = 'database.json'
    database = {
        'factory': WSJ8_kHz_FullOverlap,
    }
    pt.Configurable.get_config(database)


@ex.automain
def main(json_path, database, _log):
    database_config = database
    database = pt.configurable.config_to_instance(database)
    database_dict = {
        'datasets': {
            dataset_name: dict(tqdm(
                database.get_dataset(dataset_name).items(),
                desc=dataset_name,
            )) for dataset_name in database.dataset_names
        },
        'meta': {
            'config': pt.configurable.recursive_class_to_str(
                copy.deepcopy(database_config)
            ),
            'generated': datetime.datetime.now(),
        }
    }
    pb.io.dump(database_dict, json_path)
    _log.info(f'Wrote file: {json_path}')
