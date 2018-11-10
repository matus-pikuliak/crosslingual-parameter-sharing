from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.layer import Layer


class DEPLayer(Layer):

    def __init__(self, model, task, lang, cont_repr):
        Layer.__init__(self, model, task, lang, cont_repr)
        self.build_graph(cont_repr)

    def build_graph(self, cont_repr):

        with tf.variable_scope(self.task_code()):
            tag_count = len(self.model.dl.task_vocabs[self.task])

            self.desired_arcs = self.add_pair_labels(
                name='desired_arcs',
                depth=self.model.max_length+1)
            self.desired_labels = self.add_pair_labels(
                name='desired_labels',
                depth=tag_count)
            self.use_desired_arcs = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name='use_desired_arcs_flag')

            # shape = (sentence_lengths_sum x max_sentence_length+1 x hidden)
            pairs_repr = self.add_pairs(cont_repr)
            predicted_arcs_ids, uas_loss = self.add_uas(pairs_repr)
            las_loss = self.add_las(pairs_repr, predicted_arcs_ids, tag_count)
            self.loss = (uas_loss + las_loss) / 2

        self.train_op = self.model.add_train_op(self.loss, self.task_code())
        # TODO: Check DEP problems - cycles, zero roots, more than one roots

    PairLabel = namedtuple('PairLabels', ('placeholder', 'ids', 'one_hots'))

    def add_pair_labels(self, name, depth):
        # shape = (batch_size x max_sentence_length)
        placeholder = tf.placeholder(
            dtype=tf.int64,
            shape=[None, None],
            name=name)

        # shape = (sentence_lengths_sum)
        ids = tf.boolean_mask(
            tensor=placeholder,
            mask=self.model.sentence_lengths_mask)

        # shape = (sentence_lengths_sum x depth)
        one_hots = tf.one_hot(
            indices=ids,
            depth=depth)

        return self.PairLabel(placeholder, ids, one_hots)

    def add_pairs(self, cont_repr):

        root = tf.get_variable(
            name="root_vector",
            shape=[1, 1, 2 * self.config.word_lstm_size],
            dtype=tf.float32)
        root = tf.tile(
            input=root,
            multiples=[self.model.batch_size, 1, 1])

        cont_repr_with_root = tf.concat(
            values=[root, cont_repr],
            axis=1)

        tile_a = tf.tile(
            input=tf.expand_dims(cont_repr, 2),
            multiples=[1, 1, self.model.max_length + 1, 1])
        tile_b = tf.tile(
            input=tf.expand_dims(cont_repr_with_root, 1),
            multiples=[1, self.model.max_length, 1, 1])

        # shape = (batch_size, max_sentence_length, max_sentence_length+1, 4*word_lstm_size)
        pairs = tf.concat(
            values=[tile_a, tile_b],
            axis=3)

        valid_pairs = tf.boolean_mask(
            tensor=pairs,
            mask=self.model.sentence_lengths_mask)

        valid_pairs_repr = tf.layers.dense(
            inputs=valid_pairs,
            units=300,  # FIXME: config?
            activation=tf.nn.relu)

        return valid_pairs_repr

    def add_uas(self, pairs_repr):
        predicted_arcs_logits = tf.layers.dense(
            inputs=pairs_repr,
            units=1)
        # FIXME: add -1000 for impossible predictions? out of range words and pairs of same words
        predicted_arcs_logits = tf.squeeze(
            input=predicted_arcs_logits,
            axis=-1)  # must be specified because we need static tensor shape for boolean mask later
        predicted_arcs_ids = tf.argmax(
            input=predicted_arcs_logits,
            axis=-1)
        self.uas = tf.count_nonzero(
            tf.equal(
                predicted_arcs_ids,
                self.desired_arcs.ids
            ))

        uas_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.desired_arcs.one_hots,
            logits=predicted_arcs_logits)
        uas_loss = tf.reduce_mean(uas_loss)

        return predicted_arcs_ids, uas_loss

    def add_las(self, pairs_repr, predicted_arcs_ids, tag_count):

        selected_arcs_ids = tf.cond(
            pred=self.use_desired_arcs,
            true_fn=lambda: self.desired_arcs.ids,
            false_fn=lambda: predicted_arcs_ids)
        selected_arcs_mask = tf.one_hot(
            indices=selected_arcs_ids,
            depth=self.model.max_length + 1,
            on_value=True,
            off_value=False,
            dtype=tf.bool)
        selected_pairs_repr = tf.boolean_mask(
            tensor=pairs_repr,
            mask=selected_arcs_mask)

        predicted_labels_logits = tf.layers.dense(
            inputs=selected_pairs_repr,
            units=tag_count)
        predicted_labels_ids = tf.argmax(
            input=predicted_labels_logits,
            axis=-1)

        self.las = tf.count_nonzero(
            tf.logical_and(
                tf.equal(predicted_labels_ids, self.desired_labels.ids),
                tf.equal(predicted_arcs_ids, self.desired_arcs.ids)
            ))

        las_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.desired_labels.one_hots,
            logits=predicted_labels_logits)
        las_loss = tf.reduce_mean(las_loss)

        return las_loss

    def train(self, batch, dataset):
        *_, desired_labels, desired_arcs = batch

        fd = self.train_feed_dict(batch, dataset)
        fd.update({
            self.desired_arcs.placeholder: desired_arcs,
            self.desired_labels.placeholder: desired_labels,
            self.use_desired_arcs: True
        })
        self.model.sess.run(self.train_op, feed_dict=fd)

    def evaluate(self, iterator, dataset):

        results = []

        for batch in iterator:
            *_, desired_labels, desired_arcs = batch
            fd = self.test_feed_dict(batch, dataset)
            fd.update({
                self.desired_arcs.placeholder: desired_arcs,
                self.desired_labels.placeholder: desired_labels,
                self.use_desired_arcs: False
            })
            batch_results = self.model.sess.run(
                fetches=[self.loss, self.uas, self.las, self.model.total_batch_length],
                feed_dict=fd)
            results.append(batch_results)

        loss, uas, las, length = zip(*results)
        return {
            'loss': np.mean(loss),
            'uas': sum(uas)/sum(length),
            'las': sum(las)/sum(length)
        }
