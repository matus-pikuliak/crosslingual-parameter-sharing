import numpy as np

from data.dataset import Dataset


class NLIDataset(Dataset):

    def raw_sample_to_tuple(self, raw_sample):
        word_ids, char_ids = zip(*(
            (self.word_to_id(line[0]), self.word_to_char_ids(line[0]))
            for line in raw_sample
        ))
        for i, line in enumerate(raw_sample):
            if len(line) > 1:
                premise_length = i
                label_id = self.label_to_id(line[1])
                break
        return word_ids, char_ids, premise_length, label_id

    def _load(self):
        self.word_ids, self.char_ids, self.premise_length, self.label_id = zip(*self.load_samples())

    def prepare_samples_from_cache(self, ids):
        word_ids = [self.word_ids[i] for i in ids]
        char_ids = [self.char_ids[i] for i in ids]
        premise_length = [self.premise_length[i] for i in ids]
        label_id = [self.label_id[i] for i in ids]
        return self.prepare_samples(word_ids, char_ids, premise_length, label_id)

    def prepare_samples(self, word_ids, char_ids, premise_length, label_id):

        def split_and_join(lst, splits):
            return [
                f(lst[i], split)
                for i, split in enumerate(splits)
                for f in ((lambda x,y: x[:y]), (lambda x,y: x[y:]))
            ]

        word_ids, sentence_lengths = self.pad_sequences_1d(split_and_join(word_ids, premise_length))
        char_ids, word_lengths = self.pad_sequences_2d(split_and_join(char_ids, premise_length))

        return word_ids, sentence_lengths, char_ids, word_lengths, np.array(label_id)
