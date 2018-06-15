import itertools
import os
from dataset import Dataset

ROLES = ['train', 'dev', 'test']
UNK_WORD = '<unk>'
UNK_CHAR = u'\u2716'
EMPTY_CHAR = u'\u2717'


class DataManager:

    def __init__(self, tls=None, tasks=None, languages=None, config=None):

        # (task, language) tuples
        self.tls = tls if tls is not None else list(itertools.product(tasks, languages))

        if config is None:
            raise TypeError('Config can not be empty')
        self.config = config

        self.datasets = []
        for task, lang in self.tls:
            for role in ROLES:

                filename = os.path.join(self.config.data_path, task, lang, role)
                if not os.path.isfile(filename):
                    raise ValueError('File %s does not exist.' % filename)

                dt = Dataset(task, lang, role, filename)
                self.datasets.append(dt)

    def prepare(self):
        em = EmbeddingManager(self.languages, config)
        # nacitaj embeddings
        # ziskaj embedding dictionary
        # ziskaj konecny slovnik slov
        # ziskaj konecny slonik znakov
        # vytvor vystupnu embedding maticu
        # uprav samples v datasetoch - zmen na id, odstran nonemb slova, odstran nondict znaky, zamiesaj sample, nastav citaciu hlavu na zaciatok datasetu

    def tasks(self):
        return [tl[0] for tl in self.tls]

    def languages(self):
        return [tl[1] for tl in self.tls]

    def label_dictionaries(self):
        return None

    def word_dictionaries(self):
        return None

    def char_histogram(self):
        return None

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


