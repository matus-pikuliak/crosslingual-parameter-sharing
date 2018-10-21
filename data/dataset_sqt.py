import numpy as np

from data.dataset import Dataset


class SQTDataset(Dataset):

    def raw_sample_to_tuple(self, raw_sample):
        return zip(*(
            (self.word_to_id(line[0]), self.word_to_char_ids(line[0]), self.label_to_id(line[1]))
            for line in raw_sample
        ))

    def _load(self):
        self.word_ids, self.char_ids, self.label_ids = zip(*self.load_samples())

    def prepare_samples_by_ids(self, ids):
        word_ids = [self.word_ids[i] for i in ids]
        char_ids = [self.char_ids[i] for i in ids]
        label_ids = [self.label_ids[i] for i in ids]
        return self.prepare_samples(word_ids, char_ids, label_ids)

    def prepare_samples(self, word_ids, char_ids, label_ids):
        word_ids, sentence_lengths = self.pad_sequences_1d(word_ids)
        char_ids, word_lengths = self.pad_sequences_2d(char_ids)
        label_ids, _ = self.pad_sequences_1d(label_ids)
        return word_ids, sentence_lengths, char_ids, word_lengths, label_ids


