import numpy as np
import tensorflow as tf

from utils.general_utils import uneven_zip


class Layer:

    def __init__(self, model, cont_repr, task, lang):
        self.model = model
        self.config = model.config
        self.task = task
        self.lang = lang

    def build_graph(self, cont_repr):
        self.cont_repr = cont_repr
        self.add_adversarial_loss(cont_repr)
        self._build_graph()
        self.train_op, self.gradient_norm = self.add_train_op()
        self.add_unit_strength()
        self.metrics = self.add_metrics()

    def add_adversarial_loss(self, cont_repr):
        with tf.variable_scope('adversarial_training', reuse=tf.AUTO_REUSE):
            lambda_ = self.config.adversarial_lambda

            cont_repr = tf.boolean_mask(
                tensor=cont_repr,
                mask=self.model.sentence_lengths_mask)

            gradient_reversal = tf.stop_gradient((1 + lambda_) * cont_repr) - lambda_ * cont_repr

            hidden = tf.layers.dense(
                inputs=gradient_reversal,
                units=self.config.hidden_size,
                activation=tf.nn.relu)
            logits = tf.layers.dense(
                inputs=hidden,
                units=len(self.model.langs))

            one_hot_lang = tf.one_hot(
                indices=self.model.lang_id,
                depth=len(self.model.langs))
            one_hot_lang = tf.expand_dims(
                input=one_hot_lang,
                axis=0)
            one_hot_lang = tf.tile(
                input=one_hot_lang,
                multiples=[tf.shape(cont_repr)[0], 1])

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot_lang,
                logits=logits)
            self.adversarial_loss = tf.reduce_mean(loss)

    def add_unit_strength(self):

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

    def add_train_op(self):
        loss = self.loss
        if self.config.adversarial_training and len(self.model.langs) > 1:
            loss += self.adversarial_term()
        grads, vs = zip(*self.model.optimizer.compute_gradients(loss))
        gradient_norm = tf.global_norm(grads)
        if self.config.clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        return self.model.optimizer.apply_gradients(zip(grads, vs)), gradient_norm

    def adversarial_term(self):
        use_adversarial = tf.less_equal(
            tf.random.uniform(shape=[]),
            1 / self.config.adversarial_freq)
        use_adversarial = tf.cast(
            x=use_adversarial,
            dtype=tf.float32)
        return self.adversarial_loss * use_adversarial

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
            self.model.dropout: self.config.dropout
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
            feed_dict=fd)

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
                feed_dict=self.test_feed_dict(batch, dataset))

    def basic_fetches(self):
        return {
            'loss': self.loss,
            'adv_loss': self.adversarial_loss,
            'gradient_norm': self.gradient_norm,
            'length': self.model.total_batch_length,
            'unit_strength_2': self.unit_strength_2
        }

    def basic_results(self, results):
        return {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
            'gradient_norm': np.mean(results['gradient_norm']),
            'unit_strength_2': np.std(np.mean(results['unit_strength_2'], axis=0))
        }

    def task_code(self):
        if self.task == 'lmo':
            return f'{self.task}-{self.lang}'
        if self.config.task_layer_sharing:
            return self.task
        else:
            return f'{self.task}-{self.lang}'
