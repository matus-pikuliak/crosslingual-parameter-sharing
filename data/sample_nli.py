from .sample import Sample
import data.constants as constants
import numpy as np


class NLISample(Sample):

    def __init__(self, lines, dataset):
        Sample.__init__(self)
        self.words = []
        for i, line in enumerate(lines):
            word = line.split('\t')[0]
            self.words.append(word.strip())
            if '\t' in line:
                self.labels = [line.split('\t')[1].strip()]
                self.premise_length = i
        self.hypothesis_length = len(self.words) - self.premise_length
        self.dt = dataset

    def prepare(self, *args): # lang_vocab, task_vocab, char_vocab
        self.word_ids = self.prepare_word_ids(*args)
        self.char_ids = self.prepare_char_ids(*args)
        self.word_count = self.prepare_word_count(*args)
        self.char_count = self.prepare_char_count(*args)
        self.label_id = self.prepare_label_ids(*args)

    def padded(self, max_sentence_length, max_word_length):
        return np.array([
            self.pad_sequence(self.word_ids[:self.premise_length], max_sentence_length),
            self.pad_sequence(self.word_ids[self.premise_length:], max_sentence_length),
            self.pad_matrix(self.char_ids[:self.premise_length], max_word_length, max_sentence_length),
            self.pad_matrix(self.char_ids[self.premise_length:], max_word_length, max_sentence_length),
            self.premise_length,
            self.hypothesis_length,
            self.pad_sequence(self.char_count[:self.premise_length], max_sentence_length),
            self.pad_sequence(self.char_count[self.premise_length:], max_sentence_length),
            self.label_id,
        ])

    def prepare_word_count(self, lang_vocab, task_vocab, char_vocab):
        return max([self.premise_length, self.hypothesis_length])

    def prepare_label_ids(self, lang_vocab, task_vocab, char_vocab):
        return task_vocab.token_to_id[self.labels[0]]