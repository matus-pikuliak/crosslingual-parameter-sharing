from model.layer import Layer


class DatasetIterator:

    train_iterators = {}

    def __init__(self, dataset, config, layer, is_train=True):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.layer = layer

        if is_train:
            if dataset not in self.train_iterators:
                self.train_iterators[dataset] = self.new_iterator()
            self.iterator = self.train_iterators[dataset]
        else:
            self.iterator = self.new_iterator()

    def new_iterator(self):
        batch_size = self.config.batch_size if self.dataset.role == 'train' else 32  # FIXME: dynamic size for eval datasets

        if self.is_train:
            if self.dataset.size > self.config.max_dataset_cache:
                return self.dataset.train_file_generator(batch_size)
            else:
                return self.dataset.train_cache_generator(batch_size)

        else:
            limit = None
            if self.dataset.role == 'train':
                limit = 1000
            if self.config.eval_size_limit > 0:
                limit = self.config.eval_size_limit
            return self.dataset.test_file_generator(batch_size, limit)
