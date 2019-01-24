import numpy as np
import tensorflow as tf

from model.layer import Layer
import utils.tf_utils as tfu


class SQTLayer(Layer):

    def __init__(self, model, task, lang, cont_repr):
        Layer.__init__(self, model, task, lang, cont_repr)
        self.build_graph(cont_repr)

    def _build_graph(self):
        tag_count = len(self.model.dl.task_vocabs[self.task])

        with tf.variable_scope(self.task_code()):

            hidden, self.cont_repr_weights = tfu.dense_with_weights(
                inputs=self.cont_repr,
                units=self.model.config.hidden_size,
                activation=tf.nn.relu)

            # shape = (batch_size, max_sentence_length, tag_count)
            self.logits = tf.layers.dense(
                inputs=hidden,
                units=tag_count,
                name='predicted_logits')

            # shape = (batch_size, max_sentence_length)
            self.desired = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='desired_tag_ids')

            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.desired,
                sequence_lengths=self.model.sentence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)

    def add_metrics(self):
        metric_names = self.metric_names()

        metrics = tf.py_func(
            func=self.metrics_from_batch,
            inp=[
                self.logits,
                self.desired,
                self.model.sentence_lengths,
                self.transition_params
            ],
            Tout=[tf.int64 for _ in range(len(metric_names))]
        )

        return dict(zip(
            metric_names,
            metrics
        ))

    def basic_feed_dict(self, batch, dataset):
        fd = Layer.basic_feed_dict(self, batch, dataset)
        *_, desired = batch
        fd.update({
            self.desired: desired,
        })
        return fd

    def metrics_accumulator(self):
        raise NotImplementedError

    def crf_predict(self, logits, desired, lengths, transition_params):

        for log, des, len in zip(logits, desired, lengths):
            predicted, _ = tf.contrib.crf.viterbi_decode(
                score=log[:len],
                transition_params=transition_params
            )
            yield np.array(predicted), des[:len]
