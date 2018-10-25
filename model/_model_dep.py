import tensorflow as tf


class DEPModel:

    def add_dep(self, task_code):
        with tf.variable_scope(task_code):
            root = tf.get_variable("root_vector", dtype=self.type, shape=[2*self.config.word_lstm_size])  # dim
            root = tf.expand_dims(root, 0)
            root = tf.expand_dims(root, 0)
            root = tf.tile(
                root,
                [tf.shape(self.word_lstm_output)[0], 1, 1]
            )

            words = self.word_lstm_output
            words_root = tf.concat([root, words], 1)

            tile_a = tf.tile(
                tf.expand_dims(words, 2),
                [1, 1, tf.shape(words_root)[1], 1]
            )
            tile_b = tf.tile(
                tf.expand_dims(words_root, 1),
                [1, tf.shape(words)[1], 1, 1]
            )

            combinations = tf.concat([tile_a, tile_b], axis=3)
            combinations = tf.reshape(combinations, [-1, 4*self.config.word_lstm_size])

            hidden = 500

            W = tf.get_variable("W", dtype=self.type, shape=[4*self.config.word_lstm_size, hidden])
            b = tf.get_variable("b", dtype=self.type, shape=[hidden])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[hidden, 1])

            seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths), shape=[-1])

            combinations = tf.nn.tanh(tf.matmul(combinations, W) + b)
            all_combinations = combinations
            combinations = tf.matmul(combinations, W2)
            combinations = tf.reshape(combinations, [-1, tf.shape(words_root)[1]])  # (batch_size x length) x length+1 (root)

            self.arc_ids = tf.placeholder(tf.int64, shape=[None, None])  # batch size x length
            _arc_ids = tf.reshape(self.arc_ids, [-1])
            _arc_ids = tf.boolean_mask(_arc_ids, seq_mask)

            predicted_arc_ids = tf.argmax(combinations, axis=1)
            _predicted_arc_ids = tf.reshape(predicted_arc_ids, tf.shape(self.arc_ids))

            self.golden_arc_ids = tf.placeholder(tf.bool, shape=[])
            relevant_arc_ids = tf.cond(self.golden_arc_ids, lambda: self.arc_ids, lambda: _predicted_arc_ids)

            relevant_arc_ids = tf.one_hot(relevant_arc_ids, tf.shape(words_root)[1], on_value=True, off_value=False, dtype=tf.bool)
            relevant_arc_ids = tf.reshape(relevant_arc_ids, [-1])
            relevant_arcs = tf.boolean_mask(all_combinations, relevant_arc_ids)
            relevant_arcs = tf.boolean_mask(relevant_arcs, seq_mask)

            self.true_labels[task_code] = tf.placeholder(tf.int64, shape=[None, None], name="labels")
            labels = tf.reshape(self.true_labels[task_code], [-1])
            labels = tf.boolean_mask(labels, seq_mask)
            one_hot_labels = tf.one_hot(labels, len(self.dl.task_vocabs['dep']))

            W3 = tf.get_variable("W3", dtype=self.type, shape=[hidden, len(self.dl.task_vocabs['dep'])])
            b3 = tf.get_variable("b3", dtype=self.type, shape=[len(self.dl.task_vocabs['dep'])])
            predicted_arc_labels = tf.matmul(relevant_arcs, W3) + b3
            predicted_arc_labels_ids = tf.argmax(predicted_arc_labels, axis=1)
            loss_las = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot_labels,
                logits=predicted_arc_labels,
                dim=-1,
            ))


            predicted_arc_ids = tf.boolean_mask(predicted_arc_ids, seq_mask)
            uas = tf.equal(predicted_arc_ids, _arc_ids)
            las = tf.logical_and(tf.equal(predicted_arc_labels_ids, labels), uas)
            self.uas = tf.reduce_sum(tf.count_nonzero(uas))
            self.las = tf.reduce_sum(tf.count_nonzero(las))

            combinations = tf.boolean_mask(combinations, seq_mask)
            arc_one_hots = tf.one_hot(_arc_ids, tf.shape(words_root)[1])  # length+1
            loss_uas = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=arc_one_hots,
                logits=combinations,
                dim=-1,
            ))
            self.loss[task_code] = loss_las + loss_uas

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)


    def train(self, minibatch, task_code):

        word_ids, sentence_lengths, char_ids, word_lengths, label_ids, arcs = minibatch

        fd = {
            self.word_ids: word_ids,
            self.sequence_lengths: sentence_lengths,
            self.learning_rate: self.current_learning_rate(),
            self.dropout: self.config.dropout,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.arc_ids: arcs,
            self.true_labels[task_code]: label_ids,
            self.golden_arc_ids: True,
            self.language_flags[st.lang]: True
        }

        self.sess.run([self.train_op[task_code]], feed_dict=fd)

    def evaluate(self, set_iterator, task_code):
        uases = 0
        lases = 0
        size = 0
        losses = []

        for i, minibatch in enumerate(set_iterator):
            word_ids, sentence_lengths, char_ids, word_lengths, label_ids, arcs = minibatch

            fd = {
                self.word_ids: word_ids,
                self.true_labels[task_code]: label_ids,
                self.sequence_lengths: sentence_lengths,
                self.dropout: 1,
                self.word_lengths: word_lengths,
                self.char_ids: char_ids,
                self.arc_ids: arcs,
                self.golden_arc_ids: False,
                self.language_flags[dev_set.lang]: True
            }

            uas, las, loss = self.sess.run([self.uas, self.las, self.loss[task_code]], feed_dict=fd)
            uases += uas
            lases += las
            size += np.sum(sentence_lengths)
            losses.append(loss)

        output = {'uas': float(uases) / size, 'loss': np.mean(losses), 'las': float(lases) / size}
