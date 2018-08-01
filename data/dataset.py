import codecs

import numpy as np

from sample_sqt import SQTSample
from sample_dep import DEPSample
from sample_nli import NLISample
from sample_lmo import LMOSample

import utils.general_utils as utils


class Dataset():

    def __init__(self, task, lang, role, filename=None, samples=None, config=None, max_size=None):
        self.lang = lang
        self.task = task
        self.sample_class = {
            'ner': SQTSample,
            'pos': SQTSample,
            'dep': DEPSample,
            'nli': NLISample,
            'lmo': LMOSample
        }[self.task]
        self.role = role
        self.config = config
        if filename is not None:
            self.samples = self.load_file(filename, max_size)
        else:
            self.samples = samples
        self.reader = 0

    def __len__(self):
        return len(self.samples)

    def load_file(self, filename, max_size=None):
        bf = []
        samples = []

        with codecs.open(filename, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bf.append(line)
                else:
                    if self.config.min_sentence_length <= len(bf) <= self.config.max_sentence_length:
                        samples.append(self.sample_class(bf, self))
                        if len(samples) == max_size:
                            break
                    bf = []
        if len(bf) > 0:
            print "Warning: There are still words in buffer. Append newline at the end of file."
        return samples

    def lang_vocab(self, embedding_vocab=set()):
        vocab = dict()
        for sample in self.samples:
            for word in sample.words:
                word = word.lower()
                if word in embedding_vocab: # TODO: lower() bcs of MUSE embeddings
                    vocab.setdefault(word, 0)
                    vocab[word] += 1
        return vocab

    def task_vocab(self):
        vocab = set()
        for sample in self.samples:
            for label in sample.labels:
                vocab.add(label)
        return vocab

    def char_hist(self):
        hist = dict()
        for sample in self.samples:
            for word in sample.words:
                for char in word:
                    hist.setdefault(char, 0)
                    hist[char] += 1
        return hist

    def prepare(self, lang_vocab, task_vocab, char_vocab):
        for sample in self.samples:
            sample.prepare(lang_vocab, task_vocab, char_vocab)
        self.samples = np.array(self.samples)
        np.random.shuffle(self.samples)

    def get_samples(self, amount=1):
        return self.samples[:amount]

    def next_batch(self, batch_size): # If cyclic is set to true it will fill the batch to batch_size if it reaches the end of dataset
        samples = np.take(self.samples, xrange(self.reader, self.reader + batch_size), axis=0, mode='wrap')
        self.reader += batch_size
        if self.reader >= len(self.samples):
            self.reader = len(self.samples) % self.reader
            np.random.shuffle(self.samples)
        return self.final_batch(samples)

    def dev_batches(self, batch_size):
        batches = np.array_split(self.samples, xrange(batch_size, len(self.samples), batch_size))
        for j, batch in enumerate(batches):
            batches[j] = self.final_batch(batch)
        return batches

    def final_batch(self, samples):
        max_sentence_length = max([sample.word_count for sample in samples])
        max_word_length = max([max(sample.char_count) for sample in samples])
        data = np.array([
            sample.padded(max_sentence_length, max_word_length) for sample in samples
        ])

        return tuple([np.stack(data[:, i]) for i in xrange(data.shape[1])])


