import numpy as np

from data2.dataset import Dataset

class SQTDataset(Dataset):

    @profile
    def load(self):
        self.word_ids = [
            np.array([self.word_to_id(line[0]) for line in sample], dtype=np.int32)
            for sample
            in self.read_samples()
        ]
        self.label_ids = [
            np.array([self.label_to_id(line[1]) for line in sample], dtype=np.int32)
            for sample
            in self.read_samples()]
        pass