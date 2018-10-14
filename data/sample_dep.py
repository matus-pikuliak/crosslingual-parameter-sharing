from .sample import Sample
import numpy as np


class DEPSample(Sample):

    def __init__(self, lines, dataset):
        Sample.__init__(self)
        self.words = []
        self.arcs = []
        self.labels = []
        for line in lines:
            word, arc, arc_type = line.split('\t')
            self.words.append(word.strip())
            self.arcs.append(int(arc))
            self.labels.append(arc_type.strip())

    def prepare(self, *args): # lang_vocab, task_vocab, char_vocab
        self.word_ids = self.prepare_word_ids(*args)
        self.char_ids = self.prepare_char_ids(*args)
        self.word_count = self.prepare_word_count(*args)
        self.char_count = self.prepare_char_count(*args)
        self.label_ids = self.prepare_label_ids(*args)

    def padded(self, max_sentence_length, max_word_length):
        return np.array([
            self.pad_sequence(self.word_ids, max_sentence_length),
            self.pad_matrix(self.char_ids, max_word_length, max_sentence_length),
            self.pad_sequence(self.label_ids, max_sentence_length),
            self.word_count,
            self.pad_sequence(self.char_count, max_sentence_length),
            self.pad_sequence(self.arcs, max_sentence_length)
        ])
