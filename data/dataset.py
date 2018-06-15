import numpy as np
import codecs
from sample import Sample


class Dataset():

    def __init__(self, task, lang, role, filename):
        self.lang = lang
        self.task = task
        self.role = role
        self.samples = self.load_file(filename)
        self.prepared = False

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

    def word_dictionary(self):
        d = set()
        for sample in self.samples:
            for word in sample.words:
                d.add(word)
        return d

    def label_dictionary(self):
        d = set()
        for sample in self.samples:
            for label in sample.labels:
                d.add(label)
        return d

    def char_histogram(self):
        h = dict()
        for sample in self.samples:
            for word in sample.words:
                for char in word:
                    h.setdefault(char, 0)
                    h[char] += 1
        return h

    def prepare(self):
        return 1
        # kroky na pripravu
        # nastav prepared na True

    def get_sample(self):
        return self.samples[0]

    def next_batch(self, batch_size): # If cyclic is set to true it will fill the batch to batch_size if it reaches the end of dataset
        samples = np.take(self.samples, xrange(self.reader, self.reader + batch_size), axis=0, mode='wrap')
        self.reader += batch_size
        if self.reader > len(self.samples):
            self.reader = len(self.samples) % self.reader
            np.random.shuffle(self.samples)
        return self.final_batch(samples)

    def dev_batches(self, batch_size):
        batches = np.array_split(self.samples, xrange(batch_size, len(self.samples), batch_size))
        for j, batch in enumerate(batches):
            batches[j] = self.final_batch(batch)
        return batches

    def final_batch(self, samples):
        samples = np.copy(samples)
        max_length = np.max(samples[:, 2])
        for _, sample in enumerate(samples):
            sentence, labels, length, char_ids, word_lengths = sample
            padding = max_length - length
            sample[0] = np.append(sentence, np.zeros(padding))
            sample[1] = np.append(labels, np.zeros(padding))
        return (
            np.vstack(samples[:, 0]),
            np.vstack(samples[:, 1]),
            samples[:, 2],
            samples[:, 3],
            samples[:, 4]
        )
