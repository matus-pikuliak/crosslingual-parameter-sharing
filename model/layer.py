import tensorflow as tf


class Layer:

    def __init__(self, model, task, lang, *args):
        self.model = model
        self.config = model.config
        self.task = task
        self.lang = lang

    def task_code(self):
        return self.model.task_code(self.task, self.lang)

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

    def evaluate_batches(self, iterator, dataset, fetches):
        results = (
            self.model.sess.run(
                fetches=list(fetches.values()),
                feed_dict=self.test_feed_dict(batch, dataset)
            )
            for batch
            in iterator)

        return dict(zip(
            fetches.keys(),
            zip(*results)
        ))

    def basic_fetches(self):
        return {
            'loss': self.loss,
            'adv_loss': self.model.adversarial_loss,
            'global_norm': self.global_norm,
            'length': self.model.total_batch_length
        }

    def add_grad_stats(self, grads, cont_repr):
        self.grads = {var: grad for grad, var in grads}
        self.global_norm = tf.global_norm(list(self.grads.values()))

        grads = tf.gradients(
            ys=self.loss,
            xs=cont_repr)[0]
        self.cont_repr_grad = tf.boolean_mask(
            tensor=grads,
            mask=self.model.sentence_lengths_mask)
