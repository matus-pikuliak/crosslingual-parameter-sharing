import tensorflow as tf


class Layer:

    layers = {}

    def __init__(self, model, task, lang, *args):
        self.model = model
        self.task = task
        self.lang = lang

    def task_code(self):
        return self.model.task_code(self.task, self.lang)

    def sentence_mask(self):
        return tf.sequence_mask(self.model.sentence_lengths)

    def basic_feed_dict(self, minibatch, dataset):
        word_ids, sentence_lengths, char_ids, word_lengths, *_ = minibatch

        return {
            self.model.word_ids: word_ids,
            self.model.sentence_lengths: sentence_lengths,
            self.model.char_ids: char_ids,
            self.model.word_lengths: word_lengths,
            self.model.lang_flags[dataset.lang]: True
        }

    def train_feed_dict(self, minibatch, dataset):
        fd = self.basic_feed_dict(minibatch, dataset)
        fd.update({
            self.model.learning_rate: self.model.current_learning_rate(),
            self.model.dropout: self.model.config.dropout
        })
        return fd

    def test_feed_dict(self, minibatch, dataset):
        fd = self.basic_feed_dict(minibatch, dataset)
        fd.update({
            self.model.dropout: 1
        })
        return fd




