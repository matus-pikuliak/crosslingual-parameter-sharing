from data2.dataset import Dataset


class NLIDataset(Dataset):

    def create_samples(self):
        for sample in self.read_raw_samples():
            for i, line in enumerate(sample):
                if len(line) > 1:
                    premise_len = i
                    break
            premise_ids = [self.word_to_id(sample[i][0]) for i in range(premise_len)]
            premise_char_ids = [self.word_to_char_ids(sample[i][0]) for i in range(premise_len)]
            hypothesis_ids = [self.word_to_id(sample[i][0]) for i in range(premise_len, len(sample))]
            hypothesis_char_ids = [self.word_to_char_ids(sample[i][0]) for i in range(premise_len, len(sample))]
            label = self.label_to_id(sample[premise_len][1])
            yield premise_ids, premise_char_ids, hypothesis_ids, hypothesis_char_ids, label


    def load(self):
        self.premise_ids, self.premise_char_ids, \
        self.hypothesis_ids, self.hypothesis_char_ids, \
        self.label_ids = zip(*self.create_samples())
