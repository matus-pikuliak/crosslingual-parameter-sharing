import codecs

import numpy as np

from data.sample import Sample


class Dataset():

    def __init__(self, task, lang, role, filename=None, samples=None):
        self.lang = lang
        self.task = task
        self.role = role
        if filename is not None:
            self.samples = self.load_file(filename)
        else:
            self.samples = samples
        self.reader = 0

    def __len__(self):
        return len(self.samples)

    def load_file(self, filename):
        samples = []
        with codecs.open(filename, encoding='utf-8') as f:
            word_buffer = []
            label_buffer = []
            for line in f:
                if line.strip():
                    word, tag = line.split('\t')
                    word_buffer.append(word.strip())
                    label_buffer.append(tag.strip())
                else:
                    samples.append(Sample(word_buffer, label_buffer, self))
                    word_buffer = []
                    label_buffer = []
        return samples

    def lang_vocab(self, embedding_vocab=set()):
        vocab = set()
        for sample in self.samples:
            for word in sample.words:
                if word.lower() in embedding_vocab: # TODO: lower() bcs of MUSE embeddings
                    vocab.add(word.lower())
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
            sample.padded(50, 30) for sample in samples
        ])

        return tuple([np.stack(data[:, i]) for i in xrange(data.shape[1])])

    def final_samples(self):
        data = np.array([
            sample.padded(50, 30) for sample in self.samples
        ])

        return tuple([np.stack(data[:, i]) for i in xrange(data.shape[1])])
