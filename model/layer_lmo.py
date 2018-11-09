import numpy as np
import tensorflow as tf

from model.layer import Layer


class LMOLayer(Layer):

    # TODO: when iterating files, at least start at random position; deal with output vocab limitations

    def __init__(self, model, task, lang, cont_repr):
        Layer.__init__(self, model, task, lang, cont_repr)
        self.build_graph(cont_repr)

    def build_graph(self, cont_repr):

        with tf.variable_scope(self.task_code()):

            past = self.add_past(cont_repr)
            future = self.add_future(cont_repr)
            context = tf.concat([past, future], axis=2)
            predicted_word_ids = ...

            desired_word_ids = self.add_desired_word_ids()

            # TODO: context operations? some hidden layers?

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=desired_word_ids,
                logits=predicted_word_ids,
            ))

        self.train_op = self.model.add_train_op(self.loss, self.task_code())

    def add_past(self, cont_repr):
        start_vec = tf.get_variable('start_vec', shape=[self.config.word_lstm_size], dtype=self.type)
        start_vec = tf.expand_dims(start_vec, 0)
        start_vec = tf.tile(start_vec, [batch_size, 1])
        start_vec = tf.expand_dims(start_vec, 1)
        _fd, _ = tf.split(self.lstm_fw, [max_len - 1, 1], axis=1)
        _fd = tf.concat([start_vec, _fd], 1)

    def add_future(self, cont_repr):
        end_vec = tf.get_variable('end_vec', shape=[self.config.word_lstm_size], dtype=self.type)
        end_vec = tf.expand_dims(end_vec, 0)
        end_vec = tf.tile(end_vec, [batch_size, 1])
        end_vec = tf.expand_dims(end_vec, 1)
        one_hot = tf.one_hot(self.sequence_lengths - 1, max_len)
        one_hot = tf.expand_dims(one_hot, 2)
        end_vec = tf.matmul(one_hot, end_vec)

        _, _bd = tf.split(self.lstm_fw, [1, max_len - 1], axis=1)
        zeros = tf.zeros([batch_size, 1, self.config.word_lstm_size], dtype=self.type)
        _bd = tf.concat([_bd, zeros], 1)
        _bd = _bd + end_vec

    def add_desired_word_ids(self):
        # TODO:  # vocab filtering, flatten word_ids
        rp = tf.concat([_fd, _bd], axis=2)
        rp = tf.reshape(rp, [-1, 2 * self.config.word_lstm_size])
        seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths, max_len), [-1])
        rp = tf.boolean_mask(rp, seq_mask)
        _ids = tf.boolean_mask(tf.reshape(self.word_ids, [-1]), seq_mask)
        _ids = tf.where(
            tf.less(_ids, vocab_size),
            _ids,
            tf.zeros(tf.shape(_ids), dtype=tf.int32)
        )
        # if id is more than vocab size, it is set to <unk> id = 0


    def train(self, batch, dataset):
        fd = self.train_feed_dict(batch, dataset)
        self.model.sess.run(self.train_op, feed_dict=fd)

    def evaluate(self, iterator, dataset):
        results = []
        for batch in iterator:
            fd = self.test_feed_dict(batch, dataset)
            batch_results = self.model.sess.run(
                fetches=[self.loss, self.perplexity, self.model.total_batch_length],
                feed_dict=fd)
            results.append(batch_results)

        # FIXME: unify how loss is calculated (per word? per sentence? per batch?)

        loss, perplexity, length = zip(*results)
        return {
            'loss': np.mean(loss),
            'perplexity': sum(perplexity)/sum(length)
        }
