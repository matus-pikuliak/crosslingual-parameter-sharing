import numpy as np
import tensorflow as tf

from model.layer import Layer


class SQTLayer(Layer):

    def __init__(self, model, task, lang, cont_repr):
        Layer.__init__(self, model, task, lang, cont_repr)
        self.build_graph(cont_repr)

    def build_graph(self, cont_repr):
        tag_count = len(self.model.dl.task_vocabs[self.task])

        with tf.variable_scope(self.task_code()):

            hidden = tf.layers.dense(
                inputs=cont_repr,
                units=200,
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

        # Rremoving the train_op from the task variable scope makes the computational graph less weird.
        self.train_op = self.model.add_train_op(self.loss, self.task_code())

    def train(self, batch, dataset):
        *_, desired = batch
        fd = self.train_feed_dict(batch, dataset)
        fd.update({
            self.desired: desired,
        })
        self.model.sess.run(self.train_op, feed_dict=fd)

    def evaluate(self, iterator, dataset):

        losses = []
        accumulator = self.metrics_accumulator()
        next(accumulator)

        for batch in iterator:
            predictor = self.predict_crf_batch(batch, dataset)
            losses.append(next(predictor))
            for predicted, desired in predictor:
                metrics = accumulator.send((predicted, desired))

        metrics['loss'] = np.mean(losses)
        return metrics

    def metrics_accumulator(self):
        raise NotImplementedError

    def predict_crf_batch(self, batch, dataset):
        '''
        First yields loss and then iterate over samples in batch.
        Returned sequences are not padded anymore.
        '''
        _, sentence_lengths, _, _, desired = batch
        fd = self.test_feed_dict(batch, dataset)
        fd.update({
            self.desired: desired
        })

        logits, transition_params, loss = self.model.sess.run(
            fetches=[self.logits, self.transition_params, self.loss],
            feed_dict=fd)

        yield loss

        for log, des, len in zip(logits, desired, sentence_lengths):
            predicted, _ = tf.contrib.crf.viterbi_decode(
                score=log[:len],
                transition_params=transition_params
            )
            yield np.array(predicted), des[:len]
