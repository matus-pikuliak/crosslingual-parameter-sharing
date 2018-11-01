import numpy as np
import tensorflow as tf

from model.layer import Layer


class SQTLayer(Layer):

    def __init__(self, model, task, lang, cont_repr):

        task_code = model.task_code(task, lang)
        self.model = model

        with tf.variable_scope(task_code):
            tag_count = len(model.dl.task_vocabs[task])
            max_length = tf.shape(model.word_ids)[1]

            output = tf.reshape(cont_repr, [-1, 2 * model.config.word_lstm_size])
            W = tf.get_variable(dtype=model.type, shape=[2 * model.config.word_lstm_size, tag_count],
                                name="weights")
            b = tf.get_variable(dtype=model.type, shape=[tag_count], initializer=tf.zeros_initializer(),
                                name="biases")
            output = tf.matmul(output, W) + b
            self.predicted_labels = tf.reshape(output, [-1, max_length, tag_count])
            #FIXME: no softmax or activation function?

            # expected output
            # shape = (batch_size, max_length)
            self.true_labels = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")

            # loss
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.predicted_labels,
                self.true_labels,
                model.sentence_length)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)

        # training
        # Rremoving this part from task scope lets the graph reuse optimizer parameters
        # FIXME: Is that so?
        self.train_op = model.add_train_op(self.loss, task_code)

    def train(self, minibatch, dataset):
        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch

        fd = {
            self.model.word_ids: word_ids,
            self.true_labels: label_ids,
            self.model.sentence_length: sentence_lengths,
            self.model.learning_rate: self.model.current_learning_rate(),
            self.model.dropout: self.model.config.dropout,
            self.model.word_lengths: word_lengths,
            self.model.char_ids: char_ids,
            self.model.lang_flags[dataset.lang]: True
        }

        _, train_loss = self.model.sess.run(
            [self.train_op, self.loss]
            , feed_dict=fd
        )

    def evaluate(self, iterator, dataset):

        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0

        for i, minibatch in enumerate(iterator):
            _, sentence_lengths, _, _, label_ids = minibatch

            labels_ids_predictions, loss = self.predict_crf_batch(minibatch, dataset.lang)
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


    def predict_crf_batch(self, minibatch, lang):

        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch
        fd = {
            self.model.word_ids: word_ids,
            self.true_labels: label_ids,
            self.model.sentence_length: sentence_lengths,
            self.model.dropout: 1,
            self.model.word_lengths: word_lengths,
            self.model.char_ids: char_ids,
            self.model.lang_flags[lang]: True
        }

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params, loss = self.model.sess.run(
            [self.predicted_labels, self.trans_params, self.loss], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sentence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, loss