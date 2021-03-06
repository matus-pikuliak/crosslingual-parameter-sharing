import os
import pickle

import numpy as np

from .multithreading import multithreading
import constants

class Dataset:

    @staticmethod
    def create(task, *args):

        def subclass_name(task):
            return f'{task.upper()}Dataset'

        # loads relevant subclasses defined in task_models
        dataset_classes = {
            task: __import__(f'data.dataset_{task}', fromlist=[subclass_name(task)])
            for task
            in constants.TASKS
        }

        # creates suitable Dataset subclass based on the task user specified
        try:
            subclass = getattr(dataset_classes[task], subclass_name(task))
        except KeyError:
            raise AttributeError(f'Unknown task: {task}')
        return subclass(task, *args)

    def __init__(self, task, lang, role, config, data_loader):
        self.task = task
        self.lang = lang
        self.role = role
        self.config = config
        self.filename = os.path.join(config.data_path, task, lang, role)
        self.dl = data_loader
        self.limit = self.set_limit()
        self.loaded = False

    def __str__(self):
        return f'Dataset {self.task} {self.lang} {self.role} ({self.__class__.__name__}): {self.size}'

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
            (
                self.config.limited_task_language and
                self.config.limited_task_language.split('-') == [self.task, self.lang]
            )
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
        print(f'Loading {self}.')
        self._load()
        self.loaded = True
        print(f'Loaded.')

    def load_hists(self):
        if self.limit:
            dump = self._load_hists()
        else:
            if os.path.isfile(self.vocab_file()):
                with open(self.vocab_file(), 'rb') as pickle_file:
                    dump = pickle.load(pickle_file)

            else:
                dump = self._load_hists()
                with open(self.vocab_file(), 'wb') as pickle_file:
                    pickle.dump(dump, pickle_file)

        self.lang_hist,\
        self.char_hist,\
        self.task_hist,\
        self.size = dump

    def vocab_file(self):
        return f'{self.filename}.vocab.pckl'

    def _load_hists(self):
        lang_hist = {}
        char_hist = {}
        task_hist = {}

        counter = 0
        for sample in self.read_raw_samples():
            counter += 1
            for line in sample:

                if len(line) > 0:
                    word = line[0]
                    lang_hist.setdefault(word, 0)
                    lang_hist[word] += 1
                    for char in word:
                        char_hist.setdefault(char, 0)
                        char_hist[char] += 1

                if len(line) > 1:
                    label = line[1]
                    task_hist.setdefault(label, 0)
                    task_hist[label] += 1
        size = counter

        return lang_hist, char_hist, task_hist, size

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

        def yield_lines(lines):
            if self.config.min_sentence_length <= len(lines) <= self.config.max_sentence_length:
                yield lines

        with open(self.filename, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line.split())
                else:
                    yield from yield_lines(lines)
                    lines = []
            yield from yield_lines(lines)  # if there is anything in `lines` (no newline at the end of file)

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


    """
    train iterators are endless, test iterators iterate over dataset once.
    cache iterators work with data loaded into memory, file iterators create samples dynamically from files.
    """

    @multithreading
    def train_cache_generator(self, batch_size):
        if not self.loaded:
            self.load()
        while True:
            ids = np.random.permutation(len(self))
            for i in range(0, len(self), batch_size):
                batch_ids = ids[i:i+batch_size]
                if len(batch_ids) < batch_size:
                    break
                yield self.prepare_samples_from_cache(batch_ids)

    @multithreading
    def test_cache_generator(self, batch_size, limit=None):
        if not self.loaded:
            self.load()
        size = limit if limit and limit < len(self) else len(self)
        for i in range(0, size, batch_size):
            batch_ids = range(i, min(size, i + batch_size))
            yield self.prepare_samples_from_cache(batch_ids)

    @multithreading
    def train_file_generator(self, batch_size):
        while True:
            sample_generator = self.read_raw_samples()
            try:
                while True:
                    samples = [next(sample_generator) for _ in range(batch_size)]
                    yield self.prepare_from_raw_samples(samples)
            except StopIteration:
                pass

    @multithreading
    def test_file_generator(self, batch_size, limit=None):
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
            matrix[i, :len(seq)] = seq
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



