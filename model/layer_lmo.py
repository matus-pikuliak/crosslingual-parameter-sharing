import numpy as np
import tensorflow as tf

from model.layer import Layer


class LMOLayer(Layer):

    def __init__(self, model, task, lang, cont_repr):
        Layer.__init__(self, model, task, lang, cont_repr)
        self.build_graph(cont_repr)

    def build_graph(self, cont_repr):

        with tf.variable_scope(self.task_code()):

            past = self.add_past(cont_repr)
            future = self.add_future(cont_repr)
            # shape = (sentence_lengths_sum x 2*word_lstm_size)
            context = tf.concat([past, future], axis=1)

            hidden = tf.layers.dense(
                inputs=context,
                units=self.model.config.hidden_size,
                activation=tf.nn.relu)

            predicted_word_logits = tf.layers.dense(
                inputs=hidden,
                units=self.vocab_size())

            desired_word_ids = self.add_desired_word_ids()

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=desired_word_ids,
                logits=predicted_word_logits)
            self.perplexity = tf.reduce_sum(loss)
            self.loss = tf.reduce_mean(loss)

        self.train_op = self.model.add_train_op(self.loss)

    def vocab_size(self):
        return min(
            len(self.model.dl.lang_vocabs[self.lang]),
            self.model.config.lmo_vocab_limit
        )

    def add_past(self, cont_repr):

        start_tag = tf.get_variable(
            name='start_tag',
            shape=[1, 1, self.config.word_lstm_size],
            dtype=tf.float32)
        start_tag = tf.tile(
            input=start_tag,
            multiples=[self.model.batch_size, 1, 1])

        fw_repr, _ = tf.split(
            value=cont_repr,
            num_or_size_splits=2,
            axis=2)
        fw_repr, _ = tf.split(
            value=fw_repr,
            num_or_size_splits=[-1, 1],
            axis=1)

        fw_repr = tf.concat(
            values=[start_tag, fw_repr],
            axis=1)
        fw_repr = tf.boolean_mask(
            tensor=fw_repr,
            mask=self.model.sentence_lengths_mask)
        return fw_repr

    def add_future(self, cont_repr):
        end_tag = tf.get_variable(
            name='end_tag',
            shape=[1, 1, self.config.word_lstm_size],
            dtype=tf.float32)
        end_tag = tf.tile(
            input=end_tag,
            multiples=[self.model.batch_size, self.model.max_length, 1])

        _, bw_repr = tf.split(
            value=cont_repr,
            num_or_size_splits=2,
            axis=2)
        bw_repr = tf.roll(
            input=bw_repr,
            shift=-1,
            axis=1)

        mask = tf.sequence_mask(
            lengths=self.model.sentence_lengths - 1,
            maxlen=self.model.max_length)
        mask = tf.expand_dims(
            input=mask,
            axis=-1)
        mask = tf.tile(
            input=mask,
            multiples=[1, 1, self.config.word_lstm_size])
        bw_repr = tf.where(
            condition=mask,
            x=bw_repr,
            y=end_tag)

        bw_repr = tf.boolean_mask(
            tensor=bw_repr,
            mask=self.model.sentence_lengths_mask)
        return bw_repr

    def add_desired_word_ids(self):
        word_ids = tf.boolean_mask(
            tensor=self.model.word_ids,
            mask=self.model.sentence_lengths_mask)
        word_ids = tf.where(  # adds zero id for <UNK> instead of out of LMO vocab words
            condition=tf.less(word_ids, self.vocab_size()),
            x=word_ids,
            y=tf.zeros(
                shape=tf.shape(word_ids),
                dtype=tf.int32))
        return word_ids

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

        loss, perplexity, length = zip(*results)
        return {
            'loss': np.mean(loss),
            'perplexity': np.exp(sum(perplexity)/sum(length))
        }
