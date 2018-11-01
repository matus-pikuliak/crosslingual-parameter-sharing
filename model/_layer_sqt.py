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

    def train(self, minibatch, dataset):
        *_, desired = minibatch
        fd = self.train_feed_dict(minibatch, dataset)
        fd.update({
            self.desired: desired,
        })
        self.model.sess.run(self.train_op, feed_dict=fd)

    def evaluate(self, iterator, dataset):

        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0

        for i, minibatch in enumerate(iterator):
            _, sentence_lengths, _, _, label_ids = minibatch

            labels_ids_predictions, loss = self.predict_crf_batch(minibatch, dataset)
            losses.append(loss)

            for lab, lab_pred, length in zip(label_ids, labels_ids_predictions, sentence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                for (true_t, pred_t) in zip(lab, lab_pred):
                    accs.append(true_t == pred_t)
                    if dataset.task == 'ner':
                        O_token = self.model.dl.task_vocabs['ner'].token_to_id['O']
                        if true_t != O_token:
                            expected_ner += 1
                        if pred_t != O_token:
                            predicted_ner += 1
                        if true_t != O_token and true_t == pred_t:
                            precision += 1
                        if pred_t != O_token and true_t == pred_t:
                            recall += 1

        output = {'acc': 100 * np.mean(accs), 'loss': np.mean(losses)}
        if dataset.task == 'ner':
            precision = float(precision) / (predicted_ner + 1)
            recall = float(recall) / (expected_ner + 1)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            output.update({
                'expected_ner_count': expected_ner,
                'predicted_ner_count': predicted_ner,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        return output

    def predict_crf_batch(self, minibatch, dataset):
        _, sentence_lengths, _, _, desired = minibatch
        fd = self.test_feed_dict(minibatch, dataset)
        fd.update({
            self.desired: desired
        })

        logits, transition_params, loss = self.model.sess.run(
            fetches=[self.logits, self.transition_params, self.loss],
            feed_dict=fd)

        yield loss

        for log, len in zip(logits, sentence_lengths):
            log = log[:len]
            predicted, _ = tf.contrib.crf.viterbi_decode(
                score=log,
                transition_params=transition_params
            )
            yield predicted, desired