import random
import numpy as np


class Dataset():

    def __init__(self, language, task, role, samples):
        self.language = language
        self.task = task
        self.role = role
        self.samples = samples
        self.reader = 0
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

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
            sentence, labels, length = sample
            padding = max_length - length
            sample[0] = np.append(sentence, np.zeros(padding))
            sample[1] = np.append(labels, np.zeros(padding))
        return (np.vstack(samples[:, 0]),
                np.vstack(samples[:, 1]),
                samples[:, 2])




