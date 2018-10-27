import tensorflow as tf


class NLIModel:

    def add_task_layer(self, task_code):
        with tf.variable_scope(task_code):
            words = self.word_lstm_output
            reduction = tf.concat([
                tf.reduce_max(words, axis=1),
                tf.reduce_mean(words, axis=1)
            ], axis=1)
            reduction = tf.reshape(reduction, [-1, 8 * self.config.word_lstm_size])
            premise, hypothesis = tf.split(reduction, [4 * self.config.word_lstm_size, 4 * self.config.word_lstm_size], axis=1)
            representation = tf.concat([
                tf.multiply(premise, hypothesis),
                premise - hypothesis
            ], axis=1)
            W = tf.get_variable("W", dtype=self.type, shape=[8 * self.config.word_lstm_size, 500])
            b = tf.get_variable("b", dtype=self.type, shape=[500])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[500, len(self.dl.task_vocabs['nli'])])

            representation = tf.matmul(representation, W) + b
            representation = tf.matmul(tf.nn.tanh(representation), W2)

            self.true_labels[task_code] = tf.placeholder(tf.int64, shape=[None],
                                                  name="labels")
            true_labels_one_hot = tf.one_hot(self.true_labels[task_code], depth=len(self.dl.task_vocabs['nli']))
            self.loss[task_code] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=true_labels_one_hot,
                logits=representation,
                dim=-1,
            ))

            predicted_labels = tf.argmax(representation, axis=1)
            correct_labels = tf.equal(predicted_labels, self.true_labels[task_code])
            self.correct_labels_count = tf.reduce_sum(tf.cast(correct_labels, dtype=tf.int32))

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def train(self, minibatch, task_code):

        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch

        fd = {
            self.word_ids: word_ids,
            self.sequence_lengths: sentence_lengths,
            self.learning_rate: self.current_learning_rate(),
            self.dropout: self.config.dropout,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.true_labels[task_code]: label_ids,
            self.language_flags[st.lang]: True
        }

        self.sess.run([self.train_op[task_code]], feed_dict=fd)

    def evaluate(self, set_iterator, task_code):
        counts = []
        losses = []
        sum = 0

        for i, minibatch in enumerate(set_iterator):
            word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch

            fd = {
                self.word_ids: word_ids,
                self.true_labels[task_code]: label_ids,
                self.sequence_lengths: sentence_lengths,
                self.dropout: 1,
                self.word_lengths: word_lengths,
                self.char_ids: char_ids,
                self.language_flags[dev_set.lang]: True
            }

            count, loss = self.sess.run([self.correct_labels_count, self.loss[task_code]], feed_dict=fd)
            counts.append(count)
            losses.append(loss)
            sum += len(label_ids)

        output = {'acc': float(np.sum(counts)) / sum, 'loss': np.mean(losses)}
