import numpy as np

import constants
from data2.word_normalization import word_normalization


class BidirVocab:

    def __init__(self, tokens):
        self.t2id = {word: i for i, word in enumerate(tokens)}
        self.id2t = {i:word for i, word in enumerate(tokens)}

    def __len__(self):
        return len(self.t2id)

    def __repr__(self):
        return str({k: self.id2t[k] for k in range(10)})+f' and {len(self)-10} others'

    def __contains__(self, token):
        return token in self.t2id


class LangVocab(BidirVocab):

    def __init__(self, tokens):
        tokens.append(constants.UNK_WORD)
        BidirVocab.__init__(self, tokens)

    def word_to_id(self, word):
        word = word_normalization(word)
        return self.t2id.get(word, self.t2id[constants.UNK_WORD])


class TaskVocab(BidirVocab):

    def label_to_id(self, label):
        return self.t2id[label]


class CharVocab(BidirVocab):

    def __init__(self, chars):
        chars.append(constants.UNK_CHAR)
        BidirVocab.__init__(self, chars)

    def char_to_id(self, char):
        return self.t2id.get(char, self.t2id[constants.UNK_CHAR])

    def word_to_ids(self, word):
        return [self.char_to_id(char) for char in word]