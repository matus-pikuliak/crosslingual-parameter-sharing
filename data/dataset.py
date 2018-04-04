import random
import numpy as np


class Dataset():

    def __init__(self, language, task, role, samples):
        self.language = language
        self.task = task
        self.role = role
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self):
        return self.samples[0]

    def minibatches(self, batch_size):
        np.random.shuffle(self.samples)
        batches = np.array_split(self.samples, xrange(batch_size, len(self.samples), batch_size))
        for j, batch in enumerate(batches):
            max_length = np.max(batch[:, 2])
            for i, sample in enumerate(batch):
                sentence, labels, length = sample
                padding = max_length - length
                sample[0] = np.append(sentence, np.zeros(padding))
                sample[1] = np.append(labels, np.zeros(padding))
            batches[j] = (np.vstack(batch[:, 0]),
                          np.vstack(batch[:, 1]),
                          batch[:, 2])
        return batches



