import tensorflow as tf


class SQTModel:

    def add_crf(self, task, task_code):
        with tf.variable_scope(task_code):
            tag_count = len(self.dl.task_vocabs[task])
            max_length = tf.shape(self.word_ids)[1]

            output = tf.reshape(self.word_lstm_output, [-1, 2 * self.config.word_lstm_size])
            W = tf.get_variable(dtype=self.type, shape=[2 * self.config.word_lstm_size, tag_count],
                                name="weights")
            b = tf.get_variable(dtype=self.type, shape=[tag_count], initializer=tf.zeros_initializer(),
                                name="biases")
            output = tf.matmul(output, W) + b
            self.predicted_labels[task_code] = tf.reshape(output, [-1, max_length, tag_count])

            # expected output
            # shape = (batch_size, max_length)
            self.true_labels[task_code] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")

            # loss
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.predicted_labels[task_code],
                self.true_labels[task_code],
                self.sequence_lengths)
            self.trans_params[task_code] = trans_params  # need to evaluate it for decoding
            self.loss[task_code] = tf.reduce_mean(-log_likelihood)

        # training
        # Rremoving this part from task scope lets the graph reuse optimizer parameters
        # FIXME: Is that so?
        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def train(self, minibatch, task_code):
        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch

        fd = {
            self.word_ids: word_ids,
            self.true_labels[task_code]: label_ids,
            self.sequence_lengths: sentence_lengths,
            self.learning_rate: self.current_learning_rate(),
            self.dropout: self.config.dropout,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.language_flags[st.lang]: True
        }

        _, train_loss, gradient_norm = self.sess.run(
            [self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]]
            , feed_dict=fd
        )

    def evaluate(self, set_iterator, task_code):

        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0

        for i, minibatch in enumerate(set_iterator):
            _, sentence_lengths, _, _, label_ids = minibatch

            labels_ids_predictions, loss = self.predict_crf_batch(minibatch, task_code, dev_set.lang)
            losses.append(loss)

            for lab, lab_pred, length in zip(label_ids, labels_ids_predictions, sentence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                for (true_t, pred_t) in zip(lab, lab_pred):
                    accs.append(true_t == pred_t)
                    if dev_set.task == 'ner':
                        O_token = self.dl.task_vocabs['ner'].token_to_id['O']
                        if true_t != O_token:
                            expected_ner += 1
                        if pred_t != O_token:
                            predicted_ner += 1
                        if true_t != O_token and true_t == pred_t:
                            precision += 1
                        if pred_t != O_token and true_t == pred_t:
                            recall += 1

        output = {'acc': 100 * np.mean(accs), 'loss': np.mean(losses)}
        if dev_set.task == 'ner':
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


    def predict_crf_batch(self, minibatch, task_code, lang):

        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch
        fd = {
            self.word_ids: word_ids,
            self.true_labels[task_code]: label_ids,
            self.sequence_lengths: sentence_lengths,
            self.dropout: 1,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.language_flags[lang]: True
        }

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params, loss = self.sess.run(
            [self.predicted_labels[task_code], self.trans_params[task_code], self.loss[task_code]], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sentence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, loss