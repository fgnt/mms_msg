import lazy_dataset.database
from mms_msg.databases.utils import get_dataset_name_and_rng


class MMSMSGDatabase(lazy_dataset.database.Database):
    def _get_dataset(self, name=None):
        if not isinstance(name, str):
            return super()._get_dataset(name)
        else:
            name, rng = get_dataset_name_and_rng(name)
            return self.get_mixture_dataset(name, rng)

    def get_mixture_dataset(self, name, rng):
        raise NotImplementedError()

    def load_example(self, example):
        raise NotImplementedError()
