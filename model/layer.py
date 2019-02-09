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
        self.train_op, self.gradient_norm = self.model.add_train_op(self.loss)

        repr = tf.boolean_mask(
            tensor=self.cont_repr,
            mask=self.model.sentence_lengths_mask)

        norms = tf.norm(
            tensor=self.cont_repr_weights,
            axis=1)
        norms = tf.expand_dims(
            input=norms,
            axis=0)
        norms = tf.tile(
            input=norms,
            multiples=[self.model.total_batch_length, 1])

        norms = tf.abs(tf.multiply(norms, repr))
        norms = tf.divide(
            norms,
            tf.reduce_sum(
                input_tensor=norms,
                axis=1,
                keepdims=True
            ))
        self.unit_strength_2 = tf.reduce_mean(
            input_tensor=norms,
            axis=0)

    def basic_feed_dict(self, batch, dataset):
        word_ids, sentence_lengths, char_ids, word_lengths, *_ = batch

        return {
            self.model.word_ids: word_ids,
            self.model.sentence_lengths: sentence_lengths,
            self.model.char_ids: char_ids,
            self.model.word_lengths: word_lengths,
            self.model.lang_id: self.model.langs.index(dataset.lang),
            self.model.task_id: self.model.tasks.index(dataset.task)
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
        self.model.sess.run(
            fetches=self.train_op,
            feed_dict=fd,
            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

    def evaluate(self, iterator, dataset):

        fetches = self.basic_fetches()
        fetches.update(self.metrics)
        results = self.evaluate_batches(iterator, dataset, fetches)

        output = self.basic_results(results)
        output.update(self.evaluate_task(results))

        return output

    def evaluate_batches(self, iterator, dataset, fetches):
        results = self.evaluate_batches_iterator(
            iterator=iterator,
            dataset=dataset,
            fetch_nodes=list(fetches.values()))

        return dict(zip(
            fetches.keys(),
            zip(*results)
        ))

    def evaluate_batches_iterator(self, iterator, dataset, fetch_nodes):
        for batch in iterator:
            yield self.model.sess.run(
                fetches=fetch_nodes,
                feed_dict=self.test_feed_dict(batch, dataset),
                options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

    def basic_fetches(self):
        return {
            'loss': self.loss,
            'adv_loss': self.model.adversarial_loss,
            'gradient_norm': self.gradient_norm,
            'length': self.model.total_batch_length,
            'unit_strength_2': self.unit_strength_2
        }

    def basic_results(self, results):
        output = {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
            'gradient_norm': np.mean(results['gradient_norm']),
            'unit_strength_2': np.std(np.mean(results['unit_strength_2'], axis=0))
        }

        return output
