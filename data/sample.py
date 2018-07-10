import numpy as np
import constants


class Sample:

    def __init__(self):
        pass

    def __len__(self):
        return len(self.words)

    def pad_sequence(self, seq, width):
        return np.pad(seq, (0, width - len(seq)), mode='constant', constant_values=0)

    def pad_matrix(self, matrix, width, height):
        current_height = len(matrix)
        m = np.stack([self.pad_sequence(row, width) for row in matrix])
        m = np.append(m, np.zeros((height - current_height, width)), axis=0)
        return m

    def prepare_word_ids(self, lang_vocab, task_vocab, char_vocab):
        word_ids = []
        for word in self.words:
            word = word.lower()  # TODO: Opat kvoli embeddingom
            if word not in lang_vocab.token_to_id:
                word = constants.UNK_WORD
            word_ids.append(lang_vocab.token_to_id[word])
        return np.array(word_ids)

    def prepare_char_ids(self, lang_vocab, task_vocab, char_vocab):
        char_ids = []
        for word in self.words:
            _char_ids = []
            for char in word:
                if char not in char_vocab.token_to_id:
                    char = constants.UNK_CHAR
                _char_ids.append(char_vocab.token_to_id[char])
            char_ids.append(np.array(_char_ids))
        return np.array(char_ids)

    def prepare_word_count(self, lang_vocab, task_vocab, char_vocab):
        return len(self.words)

    def prepare_char_count(self, lang_vocab, task_vocab, char_vocab):
        return np.array([len(word) for word in self.words])

    def prepare_label_ids(self, lang_vocab, task_vocab, char_vocab):
        return np.array([task_vocab.token_to_id[label] for label in self.labels])
