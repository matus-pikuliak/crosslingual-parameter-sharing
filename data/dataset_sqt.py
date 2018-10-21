import numpy as np

from data.dataset import Dataset


class SQTDataset(Dataset):

    def raw_sample_to_tuple(self, sample):
        return zip(*(
            (self.word_to_id(line[0]), self.word_to_char_ids(line[0]), self.label_to_id(line[1]))
            for line in sample
        ))

    def create_samples(self):
        for sample in self.read_raw_samples():
            yield self.raw_sample_to_tuple(sample)

    def load(self):
        self.word_ids, self.char_ids, self.label_ids = zip(*self.create_samples())

    def prepare_samples_by_ids(self, ids):
        word_ids, sentence_lengths = self.pad_sequences_1d([self.word_ids[i] for i in ids])
        char_ids, word_lengths = self.pad_sequences_2d([self.char_ids[i] for i in ids])
        label_ids, _ = self.pad_sequences_1d([self.label_ids[i] for i in ids])
        return word_ids, sentence_lengths, char_ids, word_lengths, label_ids
