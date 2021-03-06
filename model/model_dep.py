from collections import namedtuple

import numpy as np
import tensorflow as tf

from model.model import Model
from utils import edmonds
import utils.tf_utils as tfu


class ModelDEP(Model):

    def add_task_layer(self):

        with tf.variable_scope(self.task_layer_scope(), reuse=tf.AUTO_REUSE):
            tag_count = len(self.orch.dl.task_vocabs[self.task])

            self.n.desired_arcs = self.add_pair_labels(
                name='desired_arcs',
                depth=self.n.max_length+1)
            self.n.desired_labels = self.add_pair_labels(
                name='desired_labels',
                depth=tag_count)

            hidden, self.n.contextualized_weights = tfu.dense_with_weights(
                inputs=self.n.contextualized,
                units=self.config.hidden_size,
                activation=tf.nn.relu)

            # shape = (sentence_lengths_sum x max_sentence_length+1 x hidden)
            pairs_repr = self.add_pairs(hidden)
            pairs_repr = tf.layers.dense(
                inputs=pairs_repr,
                units=self.config.hidden_size,
                activation=tf.nn.relu)

            uas_loss, predicted_arcs_logits = self.add_uas_loss(pairs_repr)
            las_loss = self.add_las_loss(pairs_repr)
            self.n.loss = (uas_loss + las_loss) / 2

            self.add_eval_metrics(predicted_arcs_logits, pairs_repr)

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
            mask=self.n.sentence_lengths_mask)

        # shape = (sentence_lengths_sum x depth)
        one_hots = tf.one_hot(
            indices=ids,
            depth=depth)

        return self.PairLabel(placeholder, ids, one_hots)

    def add_pairs(self, cont_repr):

        root = tf.get_variable(
            name="root_vector",
            shape=[1, 1, self.config.hidden_size],
            dtype=tf.float32)
        root = tf.tile(
            input=root,
            multiples=[self.n.batch_size, 1, 1])

        cont_repr_with_root = tf.concat(
            values=[root, cont_repr],
            axis=1)

        tile_a = tf.tile(
            input=tf.expand_dims(cont_repr, 2),
            multiples=[1, 1, self.n.max_length + 1, 1])
        tile_b = tf.tile(
            input=tf.expand_dims(cont_repr_with_root, 1),
            multiples=[1, self.n.max_length, 1, 1])

        # shape = (batch_size, max_sentence_length, max_sentence_length+1, 4*word_lstm_size)
        pairs = tf.concat(
            values=[tile_a, tile_b],
            axis=3)

        valid_pairs = tf.boolean_mask(
            tensor=pairs,
            mask=self.n.sentence_lengths_mask)

        return valid_pairs

    def add_uas_loss(self, pairs_repr):
        predicted_arcs_logits = tf.layers.dense(
            inputs=pairs_repr,
            units=1)
        predicted_arcs_logits = tf.squeeze(
            input=predicted_arcs_logits,
            axis=-1)  # must be specified because we need a static tensor shape for the boolean mask later

        uas_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.n.desired_arcs.one_hots,
            logits=predicted_arcs_logits)
        uas_loss = tf.reduce_mean(uas_loss)

        return uas_loss, predicted_arcs_logits

    def add_las_loss(self, pairs_repr):
        predicted_labels_logits = self.arc_ids_to_label_logits(
            arcs_ids=self.n.desired_arcs.ids,
            pairs_repr=pairs_repr)
        las_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.n.desired_labels.one_hots,
            logits=predicted_labels_logits)
        las_loss = tf.reduce_mean(las_loss)

        return las_loss

    def add_eval_metrics(self, predicted_arcs_logits, pairs_repr):

        predicted_arcs_logits = tf.nn.softmax(predicted_arcs_logits)
        predicted_arcs_ids = tf.py_func(
            func=self.edmonds_prediction,
            inp=[predicted_arcs_logits, self.n.sentence_lengths],
            Tout=tf.int32)
        predicted_arcs_ids.set_shape(self.n.desired_arcs.ids.get_shape())

        self.n.uas = tf.count_nonzero(
            tf.equal(
                predicted_arcs_ids,
                self.n.desired_arcs.ids
            ))

        predicted_labels_logits = self.arc_ids_to_label_logits(
            arcs_ids=predicted_arcs_ids,
            pairs_repr=pairs_repr,
            reuse=True)
        predicted_labels_ids = tf.argmax(
            input=predicted_labels_logits,
            axis=-1,
            output_type=tf.int32)

        self.n.las = tf.count_nonzero(
            tf.logical_and(
                tf.equal(predicted_labels_ids, self.n.desired_labels.ids),
                tf.equal(predicted_arcs_ids, self.n.desired_arcs.ids)
            ))

    def add_metrics(self):
        return {
            'uas': self.n.uas,
            'las': self.n.las
        }

    def arc_ids_to_label_logits(self, arcs_ids, pairs_repr, reuse=False):
        tag_count = len(self.orch.dl.task_vocabs[self.task])

        selected_arcs_mask = tf.one_hot(
            indices=arcs_ids,
            depth=self.n.max_length + 1,
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

    def basic_feed_dict(self, batch):
        fd = Model.basic_feed_dict(self, batch)
        *_, desired_labels, desired_arcs = batch
        fd.update({
            self.n.desired_arcs.placeholder: desired_arcs,
            self.n.desired_labels.placeholder: desired_labels
        })
        return fd

    def evaluate_task(self, results):
        return {
            'uas': sum(results['uas'])/sum(results['length']),
            'las': sum(results['las'])/sum(results['length'])
        }

    @staticmethod
    def edmonds_prediction(logits, sentence_lengths):
        result = []
        for length in sentence_lengths:
            sentence = logits[:length]
            logits = logits[length:]
            sentence = sentence[:, :length+1]

            graph = {i: {} for i in range(length + 1)}
            root_index = np.argmax(sentence[:, 0])
            graph[0][root_index+1] = -sentence[root_index][0]
            for i in range(length):
                for j in range(length):
                    if j != i:
                        graph[i+1][j+1] = -sentence[j][i+1]

            # output format = [(head, tail), (head, tail)]
            mst = edmonds.mst(graph)

            prediction = [0 for _ in range(length)]
            for head, tail in mst:
                prediction[tail-1] = head
            result.extend(prediction)

        return np.array(result, dtype=np.int32)

    def get_latest_result(self, output):
        return output['las']

    def get_best_epoch(self):
        return max(self.previous_results, key=self.previous_results.get)

