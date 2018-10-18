from data2.dataset import Dataset


class NLIDataset(Dataset):

    def create_samples(self):
        for sample in self.read_raw_samples():
            premise = []
            hypothesis = []
            ac = premise
            for line in sample:
                ac.append(self.word_to_id(line[0]))
                if len(line) > 1:
                    ac = hypothesis
                    label = self.label_to_id(line[1])
            yield premise, label, hypothesis


    def load(self):
        self.premise_ids, \
        self.hypothesis_ids, \
        self.label_ids = zip(*self.create_samples())