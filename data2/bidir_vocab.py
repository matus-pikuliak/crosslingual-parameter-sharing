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


class LangVocab(BidirVocab):

    def __init__(self, tokens):
        tokens.append(constants.UNK_WORD)
        BidirVocab.__init__(self, tokens)

    def word_to_id(self, word):
        word = word_normalization(word)
        if word in self.t2id:
            return self.t2id[word]
        else:
            return self.t2id[constants.UNK_WORD]

class TaskVocab(BidirVocab):

    def label_to_id(self, label):
        return self.t2id[label]

class CharVocab(BidirVocab):
    ...

    def char_to_id(self, char):
        ...
