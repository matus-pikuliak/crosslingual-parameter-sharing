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
            if word not in lang_vocab.token_to_id:
                word = constants.UNK_WORD
            word_ids.append(lang_vocab.token_to_id[word])
        word_ids = np.array(word_ids)

        char_ids = []
        for word in self.words:
            _char_ids = []
            for char in word:
                if char not in char_vocab.token_to_id:
                    char = constants.UNK_CHAR
                _char_ids.append(char_vocab.token_to_id[char])
            char_ids.append(np.array(_char_ids))
        char_ids = np.array(char_ids)


        word_count = len(self.words)
        char_count = np.array([len(word) for word in self.words])
        label_ids = np.array([task_vocab.token_to_id[label] for label in self.labels]

        self.prepared_data = np.array([
            word_ids,
            char_ids,
            word_count,
            char_count,
            label_ids
        ])

    def padded(self, max_sentence_length, max_word_length):
        fsdf
