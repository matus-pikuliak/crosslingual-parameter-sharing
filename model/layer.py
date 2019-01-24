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

    def evaluate_batches_iterator(self, iterator, dataset, fetches):

        fetches = list(fetches.values())
        fetches += [self.unit_to_unit_influence, self.activation_norms]

        word_count = 0
        flag = False

        for batch in iterator:
            result = self.model.sess.run(
                fetches=fetches,
                feed_dict=self.test_feed_dict(batch, dataset)
            )
            word_count += result[fetches.index(self.model.total_batch_length)]
            if not flag and word_count > 10_000:
                flag = True
                fetches = fetches[:-2]
            yield result

    def evaluate_batches(self, iterator, dataset, fetches):

        results = self.evaluate_batches_iterator(iterator, dataset, fetches)

        keys = list(fetches.keys()) + ['unit_to_unit_influence', 'activation_norms']
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

        norms = tf.reduce_sum(
            input_tensor=activations,
            axis=1,
            keepdims=True)

        self.unit_to_unit_influence = tf.reduce_sum(
            input_tensor=tf.divide(activations, norms),
            axis=0)

        activation_norms = tf.norm(
            tensor=activations,
            ord=2,
            axis=2)

        activation_norms = tf.divide(
            activation_norms,
            tf.reduce_sum(
                input_tensor=activation_norms,
                axis=1,
                keepdims=True
            ))

        self.activation_norms = tf.reduce_sum(
            input_tensor=activation_norms,
            axis=0)
