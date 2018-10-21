import os
import threading

import numpy as np


preparates = {}
event_create = {}
event_send = {}
event_end = {}

class IteratorThread(threading.Thread):
    def __init__(self, ite):
        threading.Thread.__init__(self)
        self.ite = ite
        self._stopevent = threading.Event()

    def run(self):
        while not self._stopevent.is_set():
            try:
                preparates[self.ite] = next(self.ite)
            except StopIteration:
                preparates[self.ite] = StopIteration
            event_create[self.ite].set()
            event_send[self.ite].wait()
            event_send[self.ite].clear()

    def join(self, timeout=None):
        self._stopevent.set()
        threading.Thread.join(self)


def multithread(iterator):

    def wrapped(self, *args, **kwargs):
        ite = iterator(self, *args, **kwargs)
        self.generators.append(ite) # FIXME: Send a signal to kill generator from dataset when we finish computing, then it sends signal to its thread.
        event_create[ite] = threading.Event()
        event_send[ite] = threading.Event()
        thr = IteratorThread(ite)
        thr.start()

        while True:
            event_create[ite].wait()
            event_create[ite].clear()
            to_sent = preparates[ite]
            if to_sent is StopIteration:
                thr.join()
                break
            event_send[ite].set()
            yield to_sent

    return wrapped


class Dataset:

    def __new__(cls, task, *args):
        '''
        Dark magic that overwrites __new__ method for subclasses so they do not call this function by inheritance.
        Subclasses must be listed here to avoid circular dependencies.
        The whole point of this exercise is so we can create new datasets by calling Dataset(*args) and it will
        automatically create suitable Dataset subclass.
        '''

        tasks_model = (
            (('pos', 'ner'), 'sqt'),
            (('lmo'), 'lmo'),
            (('nli'), 'nli'),
            (('dep'), 'dep')
        )

        def module_name(model):
            return f'data.dataset_{model}'

        def subclass_name(model):
            return f'{model.upper()}Dataset'

        # loads relevant subclasses defined in tassk_models
        datasets = {model: __import__(module_name(model), fromlist=[subclass_name(model)])
                    for _, model in tasks_model}

        # overwrites subclasses' __new__ method
        def subclass_new(cls, *_, **__):
            return object.__new__(cls)

        for subclass in Dataset.__subclasses__():
            subclass.__new__ = subclass_new

        # creates suitable Dataset subclass based on the task user specified
        for tasks, model in tasks_model:
            if task in tasks:
                subclass = getattr(datasets[model], subclass_name(model))
                return subclass(task, *args)

        raise AttributeError(f'Unknown task: {task}')

    def __init__(self, task, lang, role, config, data_loader):
        self.task = task
        self.lang = lang
        self.role = role
        self.config = config
        self.filename = os.path.join(config.data_path, task, lang, role)
        self.dl = data_loader
        self.limit = self.set_limit()
        self.loaded = False
        self.generators = []

    def __str__(self):
        return f'Dataset {self.task} {self.lang} {self.role} ({self.__class__.__name__})'

    def __len__(self):
        try:
            return self.size
        except AttributeError:
            self.size = sum(1 for _ in self.read_raw_samples())
            return self.size

    def set_limit(self):
        if (
                self.config.limited_language == self.lang or
                self.config.limited_task == self.task or
                (self.config.limited_task_language and self.config.limited_task_language.split('-') == [self.task, self.lang])
        ):
            assert(self.config.limited_data_size > -1)
            return self.config.limited_data_size

    def word_to_id(self, word):
        return self.dl.lang_vocabs[self.lang].word_to_id(word)

    def label_to_id(self, label):
        return self.dl.task_vocabs[self.task].label_to_id(label)

    def word_to_char_ids(self, word):
        if len(word) > self.config.max_word_length:
            word = word[:self.config.max_word_length]
        return self.dl.char_vocab.word_to_ids(word)

    def load(self):
        self._load()
        self.loaded = True

    def load_hists(self):
        self.lang_hist = {}
        self.char_hist = {}
        self.task_hist = {}

        counter = 0
        for sample in self.read_raw_samples():
            counter += 1
            for line in sample:

                if len(line) > 0:
                    word = line[0]
                    self.lang_hist.setdefault(word, 0)
                    self.lang_hist[word] += 1
                    for char in word:
                        self.char_hist.setdefault(char, 0)
                        self.char_hist[char] += 1

                if len(line) > 1:
                    label = line[1]
                    self.task_hist.setdefault(label, 0)
                    self.task_hist[label] += 1
        #self.size = counter

    def del_hists(self):
        self.lang_hist.clear()
        self.task_hist.clear()
        self.char_hist.clear()

    def get_hist(self, hist_type):
        try:
            hist = {
                'lang': self.lang_hist,
                'char': self.char_hist,
                'task': self.task_hist
            }
        except AttributeError:
            raise RuntimeError(f'Dataset.load_hists() was not called yet: ({self})')
        return hist[hist_type]

    def raw_samples(self):
        with open(self.filename, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line.split())
                else:
                    if self.config.min_sentence_length <= len(lines) <= self.config.max_sentence_length:
                        yield lines
                    lines = []

    def read_raw_samples(self):
        raw_samples = self.raw_samples()
        if self.limit:
            for _ in range(self.limit):
                yield next(raw_samples)
        else:
            yield from raw_samples

    def load_samples(self):
        for raw_sample in self.read_raw_samples():
            yield self.raw_sample_to_tuple(raw_sample)

    def prepare_from_raw_samples(self, raw_samples):
        samples = zip(*(self.raw_sample_to_tuple(raw) for raw in raw_samples))
        return self.prepare_samples(*samples)


    '''
    train iterators are endless, test iterators iterate over dataset once.
    cache iterators work with data loaded into memory, file iterators create samples dynamically from files.
    '''

    @multithread
    def train_cache_iterator(self, batch_size):
        if not self.loaded:
            self.load()
        while True:
            ids = np.random.permutation(len(self))
            for i in range(0, len(self), batch_size):
                batch_ids = ids[i:i+batch_size]
                if len(batch_ids) < batch_size:
                    break
                yield self.prepare_samples_by_ids(batch_ids)

    @multithread
    def test_cache_iterator(self, batch_size, limit=None):
        if not self.loaded:
            self.load()
        size = limit if limit and limit < len(self) else len(self)
        for i in range(0, size, batch_size):
            batch_ids = range(i, min(size, i + batch_size))
            yield self.prepare_samples_by_ids(batch_ids)

    @multithread
    def train_file_iterator(self, batch_size):
        while True:
            sample_generator = self.read_raw_samples()
            try:
                while True:
                    samples = [next(sample_generator) for _ in range(batch_size)]
                    yield self.prepare_from_raw_samples(samples)
            except StopIteration:
                pass

    @multithread
    def test_file_iterator(self, batch_size, limit=None):
        sample_generator = self.read_raw_samples()
        size = limit if limit and limit < len(self) else len(self)
        for i in range(0, size, batch_size):
            batch_ids = range(i, min(size, i + batch_size))
            samples = [next(sample_generator) for _ in batch_ids]
            yield self.prepare_from_raw_samples(samples)

    @staticmethod
    def pad_sequences_1d(sequences):
        batch_size = len(sequences)
        sequence_size = max(len(seq) for seq in sequences)
        matrix = np.zeros((batch_size, sequence_size), dtype=np.int)
        for i, seq in enumerate(sequences):
            matrix[i,:len(seq)] = seq
        lens = np.array([len(seq) for seq in sequences])
        return matrix, lens

    @staticmethod
    def pad_sequences_2d(sequences):
        batch_size = len(sequences)
        sequence_size = max(len(seq) for seq in sequences)
        subsequence_size = max(max(len(subseq) for subseq in seq) for seq in sequences)
        matrix = np.zeros((batch_size, sequence_size, subsequence_size), dtype=np.int)
        for i, seq in enumerate(sequences):
            for j, subseq in enumerate(seq):
                matrix[i, j, :len(subseq)] = subseq
        lens, _ = Dataset.pad_sequences_1d([[len(subseq) for subseq in seq] for seq in sequences])
        return matrix, lens

# FEATURES
'''
dynamic loading for big datasets
dataset stats (when loading)
'''




