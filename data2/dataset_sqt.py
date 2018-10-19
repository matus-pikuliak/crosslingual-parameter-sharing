import numpy as np

from data2.dataset import Dataset


class SQTDataset(Dataset):

    def create_samples(self):
        for sample in self.read_raw_samples():
            yield zip(
                *(
                    (self.word_to_id(line[0]), self.label_to_id(line[1]), self.word_to_char_ids(line[0]))
                    for line in sample
                )
            )

    def load(self):
        self.word_ids, self.labels_ids, self.char_ids = zip(*self.create_samples())