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