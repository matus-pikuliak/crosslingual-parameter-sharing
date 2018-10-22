from data.dataset import Dataset


class LMODataset(Dataset):

    def raw_sample_to_tuple(self, raw_sample):
        return zip(*(
            (self.word_to_id(line[0]), self.word_to_char_ids(line[0]))
            for line in raw_sample
        ))

    def _load(self):
        self.word_ids, self.char_ids = zip(*self.load_samples())

    def prepare_samples_from_cache(self, ids):
        word_ids = [self.word_ids[i] for i in ids]
        char_ids = [self.char_ids[i] for i in ids]
        return self.prepare_samples(word_ids, char_ids)

    def prepare_samples(self, word_ids, char_ids):
        word_ids, sentence_lengths = self.pad_sequences_1d(word_ids)
        char_ids, word_lengths = self.pad_sequences_2d(char_ids)
        return word_ids, sentence_lengths, char_ids, word_lengths
