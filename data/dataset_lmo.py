import numpy as np

from data.dataset import Dataset


class LMODataset(Dataset):

    def create_samples(self):
        for sample in self.read_raw_samples():
            yield zip(
                *(
                    (self.word_to_id(line[0]), self.word_to_char_ids(line[0]))
                    for line in sample
                )
            )

    def load(self):
        self.word_ids, self.char_ids = zip(*self.create_samples())