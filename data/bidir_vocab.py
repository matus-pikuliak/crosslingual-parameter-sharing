import constants
from data.word_normalization import word_normalization

class BidirVocab:

    def __init__(self, tokens, reorder=True):
        if reorder:
            tokens = sorted(tokens)
        self.t2id = {word: i for i, word in enumerate(tokens)}
        self.id2t = {i: word for i, word in enumerate(tokens)}

    def __len__(self):
        return len(self.t2id)

    def __repr__(self):
        return str({k: self.id2t[k] for k in range(min(10, len(self)))})+f' and {max(len(self)-10, 0)} others'

    def __contains__(self, token):
        return token in self.t2id

    def __iter__(self):
        return iter(self.t2id.keys())


class LangVocab(BidirVocab):

    def __init__(self, emb_hist, min_freq):
        sorted_ = sorted(emb_hist.items(), key=lambda item: (-item[1], item[0]))
        tokens = [word for word, count in sorted_ if count >= min_freq]
        tokens.insert(0, constants.UNK_WORD)
        BidirVocab.__init__(self, tokens, reorder=False)

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