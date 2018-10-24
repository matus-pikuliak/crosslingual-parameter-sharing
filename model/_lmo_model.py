import tensorflow as tf


class LMOModel:

    def add_lmo(self, task_code, lang):
        with tf.variable_scope(task_code):
            max_len = tf.reduce_max(self.sequence_lengths)
            batch_size = tf.size(self.sequence_lengths)
            vocab_size = min(self.config.lmo_vocab_size + 1, len(self.dl.lang_vocabs[lang])) # +1 <unk>

            start_vec = tf.get_variable('start_vec', shape=[self.config.word_lstm_size], dtype=self.type)
            start_vec = tf.expand_dims(start_vec, 0)
            start_vec = tf.tile(start_vec, [batch_size, 1])
            start_vec = tf.expand_dims(start_vec, 1)
            _fd, _ = tf.split(self.lstm_fw, [max_len - 1, 1], axis=1)
            _fd = tf.concat([start_vec, _fd], 1)

            end_vec = tf.get_variable('end_vec', shape=[self.config.word_lstm_size], dtype=self.type)
            end_vec = tf.expand_dims(end_vec, 0)
            end_vec = tf.tile(end_vec, [batch_size, 1])
            end_vec = tf.expand_dims(end_vec, 1)
            one_hot = tf.one_hot(self.sequence_lengths - 1, max_len)
            one_hot = tf.expand_dims(one_hot, 2)
            end_vec = tf.matmul(one_hot, end_vec)

            _, _bd = tf.split(self.lstm_fw, [1, max_len - 1], axis=1)
            zeros = tf.zeros([batch_size, 1, self.config.word_lstm_size], dtype=self.type)
            _bd = tf.concat([_bd, zeros], 1)
            _bd = _bd + end_vec

            rp = tf.concat([_fd, _bd], axis=2)
            rp = tf.reshape(rp, [-1, 2 * self.config.word_lstm_size])
            seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths, max_len), [-1])
            rp = tf.boolean_mask(rp, seq_mask)
            _ids = tf.boolean_mask(tf.reshape(self.word_ids, [-1]), seq_mask)
            _ids = tf.where(
                tf.less(_ids, vocab_size),
                _ids,
                tf.zeros(tf.shape(_ids), dtype=tf.int32)
            )
            # if id is more than vocab size, it is set to <unk> id = 0

            W = tf.get_variable("W", dtype=self.type, shape=[2 * self.config.word_lstm_size, 500])
            b = tf.get_variable("b", dtype=self.type, shape=[500])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[500, vocab_size])

            rp = tf.matmul(rp, W) + b
            rp = tf.matmul(tf.nn.tanh(rp), W2)

            self.loss[task_code] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=_ids,
                logits=rp,
            ))

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def train(self, minibatch, task_code):
        word_ids, sentence_lengths, char_ids, word_lengths = minibatch
        fd = {
            self.word_ids: word_ids,
            self.sequence_lengths: sentence_lengths,
            self.learning_rate: self.current_learning_rate(),
            self.dropout: self.config.dropout,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.language_flags[st.lang]: True
        }
        self.sess.run([self.train_op[task_code]], feed_dict=fd)