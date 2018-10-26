from model._model_dep import DEPModel
from model._model_lmo import LMOModel
from model._model_ner import NERModel
from model._model_nli import NLIModel
from model._model_pos import POSModel


class DatasetIterator:

    train_iterators = {}

    def __init__(self, dataset, config, is_train=True):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.model = globals()[f'{dataset.task.upper()}Model']

        if is_train:
            if dataset not in self.train_iterators:
                self.train_iterators[dataset] = self.new_iterator()
            self.iterator = self.train_iterators[dataset]
        else:
            self.iterator = self.new_iterator()

    def new_iterator(self):
        batch_size = self.config.batch_size

        if self.is_train:
            if self.dataset.size > self.config.max_dataset_cache:
                return self.dataset.train_file_generator(batch_size)
            else:
                return self.dataset.train_cache_generator(batch_size)

        else:
            limit = 1000 if self.dataset.role == 'train' else None
            return self.dataset.test_file_generator(batch_size, limit)
