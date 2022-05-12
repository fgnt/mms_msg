import lazy_dataset.database
from mms_msg.databases.utils import get_dataset_name_and_rng


class MMSMSGDatabase:
    def __init__(self, source_database: lazy_dataset.database.Database):
        self.source_database = source_database

    @property
    def dataset_names(self):
        return self.source_database.dataset_names

    def get_dataset(self, name):
        if isinstance(name, str):
            name = (name,)

        datasets = []
        for name_ in name:
            name_, rng = get_dataset_name_and_rng(name_)
            datasets.append(self.get_mixture_dataset(name_, rng))

        return lazy_dataset.concatenate(*datasets)

    def get_mixture_dataset(self, name, rng):
        raise NotImplementedError()

    def load_example(self, example):
        raise NotImplementedError()
