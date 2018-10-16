import itertools
import os

import utils.general_utils as utils
import constants as constants
from data2.dataset import Dataset
from data2.bidir_vocab import LangVocab, TaskVocab, CharVocab
from data2.word_normalization import word_normalization


class DataLoader:

    def __init__(self, config):
        self.config = config
        tlrs = [(task, lang, role) for ((task, lang), role) in itertools.product(self.config.tasks, constants.ROLES)]
        self.datasets = [Dataset(*tlr, config, self) for tlr in tlrs]

    def __str__(self):
        return '\n\n'.join([str(dt) for dt in self.datasets])

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

        self.lang_vocabs = self.create_lang_vocabs()
        self.task_vocabs = {task: TaskVocab(vocab) for task, vocab in self.task_hists.items()}
        self.char_vocab = CharVocab([
            char
            for char, count
            in self.char_hist.items()
            if count > self.config.min_char_freq
        ])
        self.print_char_vocab_details()

        for dt in self.datasets:
            dt.del_hists()
            dt.load()

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

    def load_embedding_vocabs(self):
        vocabs = {}
        for lang in self.langs():
            filename = os.path.join(self.config.emb_path, lang)
            with open(filename, 'r') as f:
                vocabs[lang] = {line.split()[0]: 0 for line in f}
        return vocabs

    def create_lang_vocabs(self):

        for lang in self.langs():
            for word in self.lang_hists[lang]:
                key = word_normalization(word)
                amount = self.lang_hists[lang][word]
                try:
                    self.emb_vocabs[lang][key] += amount
                except KeyError:
                    pass
            self.print_lang_vocab_details(lang)

        filter_size = self.config.min_word_freq
        return {
            lang: LangVocab([word for word, count in self.emb_vocabs[lang].items() if count >= filter_size])
            for lang in self.langs()
        }

    def print_lang_vocab_details(self, lang):
        total_token = sum(count for count in self.lang_hists[lang].values())
        total_word = len(self.lang_hists[lang])
        filt_token = sum(
            count
            for count
            in self.emb_vocabs[lang].values()
            if count >= self.config.min_word_freq
        )
        filt_word = sum(
            count >= self.config.min_word_freq
            for count
            in self.emb_vocabs[lang].values()
        )
        word_ratio = filt_word * 100 / total_word
        token_ratio = filt_token * 100 / total_token
        print(f'{lang} vocabulary constructed.\n'
              f'It contains {filt_word} words ({word_ratio:.2f}%).\n'
              f'It covers {filt_token} tokens ({token_ratio:.2f}%).\n')

    def print_char_vocab_details(self):
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
              f'It covers {filt_occ} character occurences ({occ_ratio:.2f}%).\n')



# FEATURES
'''
max_size - min_size - samplu aj slova
dataset limited_size
dynamic loading for big datasets
train-dev sets (possibly just first 1000 from train set)
dataset stats (when loading)
when loaded statically - shuffling

print vocab example to make sure loaded data make sense

'''