import tensorflow as tf


class Layer:

    def __init__(self, model, task, lang, *args):
        self.model = model
        self.config = model.config
        self.task = task
        self.lang = lang

    def task_code(self):
        return self.model.task_code(self.task, self.lang)

    def basic_feed_dict(self, batch, dataset):
        word_ids, sentence_lengths, char_ids, word_lengths, *_ = batch

        return {
            self.model.word_ids: word_ids,
            self.model.sentence_lengths: sentence_lengths,
            self.model.char_ids: char_ids,
            self.model.word_lengths: word_lengths,
            self.model.lang_id: self.model.langs.index(dataset.lang)
        }

    def train_feed_dict(self, batch, dataset):
        fd = self.basic_feed_dict(batch, dataset)
        fd.update({
            self.model.learning_rate: self.model.current_learning_rate(),
            self.model.dropout: self.model.config.dropout
        })
        return fd

    def test_feed_dict(self, batch, dataset):
        fd = self.basic_feed_dict(batch, dataset)
        fd.update({
            self.model.dropout: 1
        })
        return fd





