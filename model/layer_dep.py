from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.layer import Layer
from utils import edmonds


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

            hidden = tf.layers.dense(
                inputs=cont_repr,
                units=self.model.config.hidden_size,
                activation=tf.nn.relu)

            # shape = (sentence_lengths_sum x max_sentence_length+1 x hidden)
            pairs_repr = self.add_pairs(hidden)
            pairs_repr = tf.layers.dense(
                inputs=pairs_repr,
                units=self.model.config.hidden_size,
                activation=tf.nn.relu)

            uas_loss, predicted_arcs_logits = self.add_uas_loss(pairs_repr)
            las_loss = self.add_las_loss(pairs_repr)
            self.loss = (uas_loss + las_loss) / 2

            self.add_eval_metrics(predicted_arcs_logits, pairs_repr)

        self.train_op = self.model.add_train_op(self.loss)

    PairLabel = namedtuple('PairLabels', ('placeholder', 'ids', 'one_hots'))

    def add_pair_labels(self, name, depth):
        # shape = (batch_size x max_sentence_length)
        placeholder = tf.placeholder(
            dtype=tf.int32,
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
            shape=[1, 1, self.model.config.hidden_size],
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

        return valid_pairs

    def add_uas_loss(self, pairs_repr):
        predicted_arcs_logits = tf.layers.dense(
            inputs=pairs_repr,
            units=1)
        predicted_arcs_logits = tf.squeeze(
            input=predicted_arcs_logits,
            axis=-1)  # must be specified because we need a static tensor shape for the boolean mask later

        uas_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.desired_arcs.one_hots,
            logits=predicted_arcs_logits)
        uas_loss = tf.reduce_mean(uas_loss)

        return uas_loss, predicted_arcs_logits

    def add_las_loss(self, pairs_repr):
        predicted_labels_logits = self.arc_ids_to_label_logits(
            arcs_ids=self.desired_arcs.ids,
            pairs_repr=pairs_repr)
        las_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.desired_labels.one_hots,
            logits=predicted_labels_logits)
        las_loss = tf.reduce_mean(las_loss)

        return las_loss

    def add_eval_metrics(self, predicted_arcs_logits, pairs_repr):

        # predicted_arcs_ids = tf.argmax(
        #     input=predicted_arcs_logits,
        #     axis=-1,
        #     output_type=tf.int32)

        predicted_arcs_ids = tf.py_func(
            func=self.edmonds_prediction,
            inp=[predicted_arcs_logits, self.model.sentence_lengths],
            Tout=tf.int32)
        predicted_arcs_ids.set_shape(self.desired_arcs.ids.get_shape())

        self.predicted_arcs_ids = predicted_arcs_ids  # FIXME: erase

        self.uas = tf.count_nonzero(
            tf.equal(
                predicted_arcs_ids,
                self.desired_arcs.ids
            ))

        predicted_labels_logits = self.arc_ids_to_label_logits(
            arcs_ids=predicted_arcs_ids,
            pairs_repr=pairs_repr,
            reuse=True)
        predicted_labels_ids = tf.argmax(
            input=predicted_labels_logits,
            axis=-1,
            output_type=tf.int32)

        self.las = tf.count_nonzero(
            tf.logical_and(
                tf.equal(predicted_labels_ids, self.desired_labels.ids),
                tf.equal(predicted_arcs_ids, self.desired_arcs.ids)
            ))

    def arc_ids_to_label_logits(self, arcs_ids, pairs_repr, reuse=False):
        tag_count = len(self.model.dl.task_vocabs[self.task])

        selected_arcs_mask = tf.one_hot(
            indices=arcs_ids,
            depth=self.model.max_length + 1,
            on_value=True,
            off_value=False,
            dtype=tf.bool)
        selected_pairs_repr = tf.boolean_mask(
            tensor=pairs_repr,
            mask=selected_arcs_mask)
        predicted_labels_logits = tf.layers.dense(
            inputs=selected_pairs_repr,
            units=tag_count,
            name=f'label_logits',
            reuse=reuse)

        return predicted_labels_logits

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
                fetches=[self.loss, self.model.adversarial_loss, self.uas, self.las, self.model.total_batch_length],
                feed_dict=fd)
            results.append(batch_results)

        loss, adv_loss, uas, las, length = zip(*results)
        return {
            'loss': np.mean(loss),
            'adv_loss': np.mean(adv_loss),
            'uas': sum(uas)/sum(length),
            'las': sum(las)/sum(length)
        }

    @staticmethod
    def edmonds_prediction(logits, sentence_lengths):
        result = []
        for length in sentence_lengths:
            sentence = logits[:length]
            logits = logits[length:]
            sentence = sentence[:, :length+1]

            graph = {i: {} for i in range(length + 1)}
            for i in range(length):
                for j in range(length + 1):
                    if j != i+1:
                        graph[j][i+1] = -sentence[i][j]

            # output format = {head_id: {dep_id: score, dep2_id: score}, ...}
            mst = edmonds.mst(graph)

            prediction = [0 for _ in range(length)]
            for arc in mst.values():
                    prediction[arc.tail-1] = arc.head
            result.extend(prediction)

        return np.array(result, dtype=np.int32)

