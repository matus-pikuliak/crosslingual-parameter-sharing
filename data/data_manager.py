import itertools
import os
import numpy as np
from dataset import Dataset
from bidir import Bidir
from embedding_manager import EmbeddingManager
import utils.general_utils as utils
import constants


class DataManager:

    def __init__(self, tls=None, tasks=None, languages=None, config=None):

        # (task, language) tuples
        self.tls = tls if tls is not None else list(itertools.product(tasks, languages))

        if config is None:
            raise TypeError('Config can not be empty')
        self.config = config

        self.datasets = []
        for task, lang in self.tls:
            for role in constants.ROLES:

                filename = os.path.join(self.config.data_path, task, lang, role)
                if not os.path.isfile(filename):
                    raise ValueError('File %s does not exist.' % filename)

                dt = Dataset(task, lang, role, filename=filename, config=self.config)
                self.datasets.append(dt)

    def print_stats(self):
        for dt in self.datasets:
            dt.print_stats()

    def prepare(self):

        # load embedding files
        em = EmbeddingManager(self.languages(), self.config)
        em.load_files()

        # generate vocabularies for languages
        self.lang_vocabs = dict()
        self.embeddings = dict()
        for lang in self.languages():
            vocab = dict()
            for dt in self.filter_datasets(lang=lang):
                dt_vocab = dt.lang_vocab(embedding_vocab=em.vocab(lang))
                for word in dt_vocab:
                    vocab.setdefault(word, 0)
                    vocab[word] += dt_vocab[word]
            vocab = sorted(vocab.items(), key=lambda item: -item[1])
            vocab = [word for (word, _) in vocab]
            vocab.insert(0, constants.UNK_WORD)
            self.embeddings[lang] = np.zeros((len(vocab), self.config.word_emb_size))
            self.lang_vocabs[lang] = Bidir(vocab)
            for id, word in self.lang_vocabs[lang].id_to_token.iteritems():
                if word is not constants.UNK_WORD:
                    self.embeddings[lang][id] = em.embedding(lang, word)

        # create vocabularies for task labels
        self.task_vocabs = dict()
        for task in self.tasks():
            vocab = set()
            for dt in self.filter_datasets(task=task):
                vocab |= dt.task_vocab()
            self.task_vocabs[task] = Bidir(vocab)

        # create vocabularies for characters used
        char_hist = dict()
        for dt in self.datasets:
            dt_hist = dt.char_hist()
            for char, freq in dt_hist.iteritems():
                char_hist.setdefault(char, 0)
                char_hist[char] += freq
        self.char_vocab = set([char for char in char_hist.keys() if char_hist[char] > 50])
        self.char_vocab.add(constants.UNK_CHAR)
        self.char_vocab.add(constants.EMPTY_CHAR)
        self.char_vocab = Bidir(self.char_vocab)

        # create test-dev datasets?
        for dt in self.filter_datasets(role="train"):
            self.datasets.append(Dataset(dt.task, dt.lang, 'train-dev', samples=dt.get_samples(1024), config=self.config))

        # create final datasets and remove fulltext information
        for dt in self.datasets:
            dt.prepare(self.lang_vocabs[dt.lang], self.task_vocabs[dt.task], self.char_vocab)

    def tasks(self):
        return set([tl[0] for tl in self.tls])

    def languages(self):
        return set([tl[1] for tl in self.tls])

    def char_count(self):
        return len(self.char_vocab)

    def fetch_dataset(self, task, lang, role):
        for dt in self.datasets:
            if (
                task == dt.task and
                lang == dt.lang and
                role == dt.role
            ):
                return dt
        raise AttributeError('No dataset with required parameters: %s %s %s' % (task, lang, role))

    def filter_datasets(self, task=None, lang=None, role=None):
        result = []
        for dt in self.datasets:
            if (
                (task is None or task == dt.task) and
                (lang is None or lang == dt.lang) and
                (role is None or role == dt.role)
            ):
                result.append(dt)
        return result


