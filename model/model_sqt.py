import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from model.model import Model


class ModelSQT(Model):

    def add_task_layer(self):
        tag_count = len(self.orch.dl.task_vocabs[self.task])

        with tf.variable_scope(self.task_layer_scope(), reuse=tf.AUTO_REUSE):

            hidden, self.n.contextualized_weights = tfu.dense_with_weights(
                inputs=self.n.contextualized,
                units=self.config.hidden_size,
                activation=tf.nn.relu)

            # shape = (batch_size, max_sentence_length, tag_count)
            self.n.logits = tf.layers.dense(
                inputs=hidden,
                units=tag_count,
                name='predicted_logits')

            # shape = (batch_size, max_sentence_length)
            self.n.desired = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='desired_tag_ids')

            log_likelihood, self.n.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.n.logits,
                tag_indices=self.n.desired,
                sequence_lengths=self.n.sentence_lengths)

            self.n.loss = tf.reduce_mean(-log_likelihood)

    def add_metrics(self):
        metric_names = self.metric_names()

        metrics = tf.py_func(
            func=self.metrics_from_batch,
            inp=[
                self.n.logits,
                self.n.desired,
                self.n.sentence_lengths,
                self.n.transition_params
            ],
            Tout=[tf.int64 for _ in range(len(metric_names))]
        )

        return dict(zip(
            metric_names,
            metrics
        ))

    def metric_names(self):
        raise NotImplementedError

    def basic_feed_dict(self, batch):
        fd = Model.basic_feed_dict(self, batch)
        *_, desired = batch
        fd.update({
            self.n.desired: desired,
        })
        return fd

    def metrics_accumulator(self):
        raise NotImplementedError

    @staticmethod
    def crf_predict(logits, desired, lengths, transition_params):

        for log, des, len in zip(logits, desired, lengths):
            predicted, _ = tf.contrib.crf.viterbi_decode(
                score=log[:len],
                transition_params=transition_params
            )
            yield np.array(predicted), des[:len]
