import numpy as np

from data2.dataset import Dataset
from utils.general_utils import profile as prf


class LMODataset(Dataset):

    @profile
    def load(self):
        prf()                           
        self.word_ids = [np.array([self.word_to_id(line[0]) for line in sample], dtype=np.int32) for sample in self.read_samples()]
        prf()
        pass