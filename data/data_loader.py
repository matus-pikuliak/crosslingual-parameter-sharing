import itertools
import os

import utils.general_utils as utils
import constants as constants
from data.dataset import Dataset
from data.bidir_vocab import LangVocab, TaskVocab, CharVocab
from data.word_normalization import word_normalization


class DataLoader:

    def __init__(self, config):
        self.config = config
        tlrs = [(task, lang, role) for ((task, lang), role) in itertools.product(self.config.tasks, constants.ROLES)]
        self.datasets = [Dataset.create(*tlr, config, self) for tlr in tlrs]

    def __str__(self):
        return 'Created:\n'+'\n'.join([str(dt) for dt in self.datasets])+'\n'

    def tasks(self):
        return {tl[0] for tl in self.config.tasks}

    def langs(self):
        return {tl[1] for tl in self.config.tasks}

    def find(self, task=None, lang=None, role=None):
        def cond(dt, task, lang, role):
            return (
                (task is None or task == dt.task) and
                (lang is None or lang == dt.lang) and
                (role is None or role == dt.role)
            )
        return [dt for dt in self.datasets if cond(dt, task, lang, role)]

    def find_one(self, *args, **kwargs):
        return self.find(*args, **kwargs)[0]

    def load(self):
        self.lang_hists, self.task_hists, self.char_hist = self.load_hists()
        self.emb_vocabs = self.load_embedding_vocabs()

        self.lang_vocabs, self.task_vocabs, self.char_vocab = self.load_vocabs()
        self.print_vocab_details()
        self.del_hists()
        print(self)

    def load_hists(self):
        for dt in self.datasets:
            dt.load_hists()

        lang_hists = {lang: self.combine_hists('lang', lang=lang) for lang in self.langs()}
        task_hists = {task: self.combine_hists('task', task=task) for task in self.tasks()}
        char_hist = self.combine_hists('char')

        return lang_hists, task_hists, char_hist

    def combine_hists(self, hist_type, *args, **kwargs):
        dts = self.find(*args, **kwargs)
        return utils.add_hists([dt.get_hist(hist_type) for dt in dts])

    def del_hists(self):
        for dt in self.datasets:
            dt.del_hists()
        del self.emb_vocabs

    def load_embedding_vocabs(self):
        vocabs = {}
        for lang in self.langs():
            filename = os.path.join(self.config.emb_path, lang)
            with open(filename, 'r') as f:
                next(f)  # Skip first line with dimensions
                vocabs[lang] = {line.split()[0]: 0 for line in f}
        return vocabs

    def load_vocabs(self):
        lang_vocabs = self.load_lang_vocabs()
        task_vocabs = self.load_task_vocabs()
        char_vocab = self.load_char_vocab()
        return lang_vocabs, task_vocabs, char_vocab

    def load_lang_vocabs(self):

        for lang in self.langs():
            for word in self.lang_hists[lang]:
                key = word_normalization(word)
                amount = self.lang_hists[lang][word]
                try:
                    self.emb_vocabs[lang][key] += amount
                except KeyError:
                    pass

        return {
            lang: LangVocab(self.emb_vocabs[lang], self.config.min_word_freq)
            for lang in self.langs()
        }

    def load_task_vocabs(self):
        return {task: TaskVocab(vocab) for task, vocab in self.task_hists.items()}

    def load_char_vocab(self):
        return CharVocab([
            char
            for char, count
            in self.char_hist.items()
            if count > self.config.min_char_freq
        ])

    def print_vocab_details(self):
        for lang in self.lang_vocabs:
            total_token = sum(count for count in self.lang_hists[lang].values())
            filt_token = sum(
                count
                for word, count
                in self.emb_vocabs[lang].items()
                if word in self.lang_vocabs[lang]
            )

            total_word = len(self.lang_hists[lang])
            filt_word = len(self.lang_vocabs[lang])

            word_ratio = filt_word * 100 / total_word
            token_ratio = filt_token * 100 / total_token
            print(f'{lang} vocabulary constructed.\n'
                  f'It contains {filt_word} words ({word_ratio:.2f}%).\n'
                  f'It covers {filt_token} tokens ({token_ratio:.2f}%).\n'
                  f'{self.lang_vocabs[lang]}\n')

        total_char = len(self.char_hist)
        total_occ = sum(count for count in self.char_hist.values())
        filt_char = sum(
            count >= self.config.min_char_freq
            for count
            in self.char_hist.values()
        )
        filt_occ = sum(
            count
            for count
            in self.char_hist.values()
            if count >= self.config.min_char_freq
        )
        char_ratio = filt_char * 100 / total_char
        occ_ratio = filt_occ * 100 / total_occ
        print(f'Character vocabulary constructed.\n'
              f'It contains {filt_char} characters ({char_ratio:.2f}%).\n'
              f'It covers {filt_occ} character occurrences ({occ_ratio:.2f}%).\n'
              f'{self.char_vocab}\n')
