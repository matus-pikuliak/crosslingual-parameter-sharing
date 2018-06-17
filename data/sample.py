import numpy as np
import constants


class Sample:

    def __init__(self, word_buffer, label_buffer, dataset):
        self.words = word_buffer
        self.labels = label_buffer
        self.dt = dataset

    def prepare(self, lang_vocab, task_vocab, char_vocab):
        word_ids = []
        for word in self.words:
            word = word.lower() # TODO: Opat kvoli embeddingom
            if word not in lang_vocab.token_to_id:
                word = constants.UNK_WORD
            word_ids.append(lang_vocab.token_to_id[word])
        self.word_ids = np.array(word_ids)

        char_ids = []
        for word in self.words:
            _char_ids = []
            for char in word:
                if char not in char_vocab.token_to_id:
                    char = constants.UNK_CHAR
                _char_ids.append(char_vocab.token_to_id[char])
            char_ids.append(np.array(_char_ids))
        self.char_ids = np.array(char_ids)

        self.word_count = len(self.words)
        self.char_count = np.array([len(word) for word in self.words])
        self.label_ids = np.array([task_vocab.token_to_id[label] for label in self.labels])

    def padded(self, max_sentence_length, max_word_length):
        return np.array([
            self.pad_sequence(self.word_ids, max_sentence_length),
            self.pad_matrix(self.char_ids, max_word_length, max_sentence_length),
            self.pad_sequence(self.label_ids, max_sentence_length),
            self.word_count,
            self.char_count
        ])

    def pad_sequence(self, seq, width):
        return np.pad(seq, (0, width - len(seq)), mode='constant', constant_values=0)

    def pad_matrix(self, matrix, width, height):
        return np.stack([self.pad_sequence(row, width) for row in matrix])
