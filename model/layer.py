import numpy as np
import tensorflow as tf

from utils.general_utils import uneven_zip


class Layer:

    def __init__(self, model, task, lang, *args):
        self.model = model
        self.config = model.config
        self.task = task
        self.lang = lang

    def task_code(self):
        return self.model.task_code(self.task, self.lang)

    def build_graph(self, cont_repr):
        self.cont_repr = cont_repr
        self._build_graph()
        self.add_output_nodes()
        self.metrics = self.add_metrics()

    def add_output_nodes(self):
        self.train_op, grads = self.model.add_train_op(self.loss)
        self.global_norm = tf.global_norm(grads)

        matrices = tf.expand_dims(
            input=self.cont_repr_weights,
            axis=0)
        matrices = tf.tile(
            input=matrices,
            multiples=[self.model.total_batch_length, 1, 1])

        repr = tf.expand_dims(
            input=self.model.cont_repr_with_mask,
            axis=-1)

        activations = tf.multiply(matrices, repr)
        activations = tf.abs(activations)

        norms = tf.reduce_sum(
            input_tensor=activations,
            axis=1,
            keepdims=True)

        unit_strength = tf.reduce_sum(
            input_tensor=tf.divide(activations, norms),
            axis=[0, 2])
        self.unit_strength = tf.divide(
            x=unit_strength,
            y=tf.cast(
                x=tf.shape(activations)[0] * tf.shape(activations)[2],
                dtype=tf.float32
            ))

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

    def train(self, batch, dataset):
        fd = self.train_feed_dict(batch, dataset)
        self.model.sess.run(self.train_op, feed_dict=fd)

    def evaluate(self, iterator, dataset):

        fetches = self.basic_fetches()
        fetches.update(self.metrics)
        results = self.evaluate_batches(iterator, dataset, fetches)

        output = self.basic_results(results)
        output.update(self.evaluate_task(results))

        return output

    def evaluate_batches_iterator(self, iterator, dataset, fetches):

        fetches = list(fetches.values())
        flag = True
        word_count = 0

        if dataset.role in ['train', 'test']:
            fetches += [self.unit_strength]
            flag = False

        for batch in iterator:
            result = self.model.sess.run(
                fetches=fetches,
                feed_dict=self.test_feed_dict(batch, dataset)
            )

            word_count += result[fetches.index(self.model.total_batch_length)]
            if not flag and word_count > 10_000:
                flag = True
                fetches = fetches[:-1]
            yield result

    def evaluate_batches(self, iterator, dataset, fetches):

        results = self.evaluate_batches_iterator(iterator, dataset, fetches)

        keys = list(fetches.keys()) + ['unit_strength']
        return dict(zip(
            keys,
            uneven_zip(*results)
        ))

    def basic_fetches(self):
        return {
            'loss': self.loss,
            'adv_loss': self.model.adversarial_loss,
            'global_norm': self.global_norm,
            'length': self.model.total_batch_length,
        }

    def basic_results(self, results):
        output = {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
        }
        try:
            output.update({
                'unit_strength': np.std(np.mean(results['unit_strength'], axis=0))
            })
        except KeyError:
            pass

        return output
