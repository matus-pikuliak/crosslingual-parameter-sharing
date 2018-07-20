import tensorflow as tf
import numpy as np
import datetime
import os
from logs.logger_init import LoggerInit


class Model:

    def __init__(self, data_manager, config):
        self.config = config
        self.dm = data_manager
        self.sess = None
        self.logger = self.initialize_logger()

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def task_code(self, task, lang):
        if self.config.crf_sharing:
            return task
        else:
            return task+lang

    def add_train_op(self, loss, task_code):
        grads, vs = zip(*self.optimizer.compute_gradients(loss))
        self.gradient_norm[task_code] = tf.global_norm(grads)
        if self.config.clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        return self.optimizer.apply_gradients(zip(grads, vs))

    def add_crf(self, task, task_code):
        with tf.variable_scope(task_code):
            tag_count = len(self.dm.task_vocabs[task])

            # expected output
            # shape = (batch%size, max_length)
            self.true_labels[task_code] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")
            # projection
            W = tf.get_variable(dtype=tf.float32, shape=[2 * self.config.word_lstm_size, tag_count],
                                name="proj_weights")
            b = tf.get_variable(dtype=tf.float32, shape=[tag_count], initializer=tf.zeros_initializer(),
                                name="proj_biases")

            max_length = tf.shape(self.word_lstm_output)[1]  # TODO: toto moze byt vstupom z vonku?
            reshaped_output = tf.reshape(self.word_lstm_output, [-1,
                                                                 2 * self.config.word_lstm_size])  # We can apply the weight on all outputs of LSTM now
            proj = tf.matmul(reshaped_output, W) + b
            self.predicted_labels[task_code] = tf.reshape(proj, [-1, max_length, tag_count])

            # loss
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.predicted_labels[task_code],
                self.true_labels[task_code],
                self.sequence_lengths)
            self.trans_params[task_code] = trans_params  # need to evaluate it for decoding
            self.loss[task_code] = tf.reduce_mean(-log_likelihood)

            # training
            # Rremoving this part from task scope lets the graph reuse optimizer parameters
            self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def add_dep(self, task, task_code):
        with tf.variable_scope(task_code):
            seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths), shape=[-1])

            root = tf.get_variable("root_vector", dtype=tf.float32, shape=[2*self.config.word_lstm_size])  # dim
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

            W = tf.get_variable("W", dtype=tf.float32, shape=[4*self.config.word_lstm_size, 500])
            b = tf.get_variable("b", dtype=tf.float32, shape=[500])
            W2 = tf.get_variable("W2", dtype=tf.float32, shape=[500, 1])

            combinations = tf.nn.tanh(tf.matmul(combinations, W) + b)
            combinations = tf.matmul(combinations, W2)
            combinations = tf.reshape(combinations, [-1, tf.shape(words_root)[1]])  # length+1 (root)
            combinations = tf.boolean_mask(combinations, seq_mask)

            self.arc_ids = tf.placeholder(tf.int64, shape=[None, None])  # batch size x length
            _arc_ids = tf.reshape(self.arc_ids, [-1])
            _arc_ids = tf.boolean_mask(_arc_ids, seq_mask)
            arc_one_hots = tf.one_hot(_arc_ids, tf.shape(words_root)[1])  # length+1

            self.dep_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=arc_one_hots,
                logits=combinations,
                dim=-1,
            ))

            self.dep_train_op = self.add_train_op(self.dep_loss, task_code)

            predicted_arc_ids = tf.argmax(combinations, axis=1)
            self.uas = 1 - tf.cast(tf.count_nonzero(predicted_arc_ids - _arc_ids), tf.float32) / tf.cast(tf.reduce_sum(self.sequence_lengths), tf.float32)


    def build_graph(self):


        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
        name="lr")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
        name="dropout")

        #
        # inputs
        # shape = (batch size, max_sentence_length)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
        name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
        name="sequence_lengths")
        # shape = (batch_size, max_sentence_length, max_word_length)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
        name="char_ids")
        # shape = (batch_size, max_sentence_length)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
        name="word_lengths")

        # optimizer
        available_optimizers = {
            'rmsprop': tf.train.RMSPropOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'adam': tf.train.AdamOptimizer,
            'sgd': tf.train.GradientDescentOptimizer
        }
        selected_optimizer = available_optimizers[self.config.optimizer]
        self.optimizer = selected_optimizer(self.learning_rate)

        #
        # word embeddings
        _word_embeddings = tf.get_variable(
            dtype=tf.float32,
            # shape=[None, self.config.word_emb_size],
            initializer=tf.cast(self.dm.embeddings, tf.float32),
            trainable=self.config.train_emb,
            name="word_embeddings")
        self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
        name="word_embeddings_lookup")

        if self.config.char_level:
            _char_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=(self.dm.char_count(), self.config.char_emb_size),
                name="char_embeddings"
            )
            self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
            name="char_embeddings_lookup")

            with tf.variable_scope("char_bi-lstm"):
                max_sentence_length = tf.shape(self.char_embeddings)[1]
                max_word_length = tf.shape(self.char_embeddings)[2]
                self.char_embeddings = tf.reshape(self.char_embeddings, [-1, max_word_length, self.config.char_emb_size], name="abc")
                self.word_lengths_seq = tf.reshape(self.word_lengths, [-1], name="bcd")
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.char_lstm_size)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.char_embeddings,
                    sequence_length=self.word_lengths_seq, dtype=tf.float32)
                # shape(batch_size*max_sentence, max_word, 2 x word_lstm_size)
                self.char_lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                self.char_lstm_output = tf.reduce_mean(self.char_lstm_output, 1)
                self.char_lstm_output = tf.reshape(self.char_lstm_output, (-1, max_sentence_length, 2*self.config.char_lstm_size))
                self.word_embeddings = tf.concat([self.word_embeddings, self.char_lstm_output], axis=-1)

        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)

        #
        # bi-lstm
        with tf.variable_scope("word_bi-lstm"):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.word_lstm_size)
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.word_lstm_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            # shape(batch_size, max_length, 2 x word_lstm_size)
            self.word_lstm_output = tf.concat([output_fw, output_bw], axis=-1)

        self.word_lstm_output = tf.nn.dropout(self.word_lstm_output, self.dropout)

        self.true_labels = dict()
        self.predicted_labels = dict()
        self.trans_params = dict()
        self.loss = dict()
        self.train_op = dict()
        self.gradient_norm = dict()

        used_task_codes = []
        for (task, lang) in self.dm.tls:
            task_code = self.task_code(task, lang)
            if task_code not in used_task_codes:
                if task in ('ner', 'pos'):
                    self.add_crf(task, task_code)
                if task in ('dep'):
                    self.add_dep(task, task_code)
                used_task_codes.append(task_code)

        if self.config.use_gpu:
            self.sess = tf.Session(config=tf.ConfigProto(
            ))
        else:
            self.sess = tf.Session(config=tf.ConfigProto(
                device_count={'GPU': 0, 'CPU': 1},
            ))
        self.sess.run(tf.global_variables_initializer())

        if self.config.show_graph:
            tf.summary.FileWriter(self.config.log_path, self.sess.graph)
            for variable in tf.global_variables():
                print variable

    def run_experiment(self, train, test, epochs):
        self.logger.log_critical('Run started.')
        self.name = ' '.join([' '.join(t) for t in train])
        self.logger.log_message("Now training " + self.name)
        start_time = datetime.datetime.now()
        self.logger.log_message(start_time)
        self.logger.log_message(self.config.dump())
        for i in xrange(epochs):
            self.run_epoch(i, train=train, test=test)
        end_time = datetime.datetime.now()
        self.logger.log_message(end_time)
        self.logger.log_message('Training took: '+str(end_time-start_time))
        self.logger.log_critical('Run done.')

    def run_epoch(self, epoch_id,
                  train,
                  test):

        train_sets = [self.dm.fetch_dataset(task, lang, 'train') for (task, lang) in train]
        dev_sets = [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in train]
        dev_sets += [self.dm.fetch_dataset(task, lang, 'train-dev') for (task, lang) in train]
        dev_sets += [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in test]

        for _ in xrange(self.config.epoch_steps):
            for st in train_sets:
                task_code = self.task_code(st.task, st.lang)

                if st.task in ('ner', 'pos'):

                    minibatch = st.next_batch(self.config.batch_size)
                    word_ids, char_ids, label_ids, sentence_lengths, word_lengths = minibatch

                    fd = {
                        self.word_ids: word_ids,
                        self.true_labels[task_code]: label_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.config.learning_rate,
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids
                    }

                    _, train_loss, gradient_norm = self.sess.run(
                        [self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]]
                        , feed_dict=fd
                    )


                if st.task in ('dep'):
                    minibatch = st.next_batch(self.config.batch_size)
                    word_ids, char_ids, label_ids, sentence_lengths, word_lengths, arcs = minibatch

                    fd = {
                        self.word_ids: word_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.config.learning_rate,
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids,
                        self.arc_ids: arcs
                    }

                    self.sess.run([self.dep_train_op], feed_dict=fd)

        self.logger.log_message("End of epoch " + str(epoch_id+1))

        for st in dev_sets:
            metrics = {
                'language': st.lang,
                'task': st.task,
                'role': st.role,
                'epoch': epoch_id + 1,
                'run': self.name
            }
            metrics.update(self.run_evaluate(st))
            self.logger.log_result(metrics)

    def run_evaluate(self, dev_set):
        task_code = self.task_code(dev_set.task, dev_set.lang)

        if dev_set.task in ('ner', 'pos'):

            accs = []
            losses = []
            expected_ner = 0
            predicted_ner = 0
            precision = 0
            recall = 0

            for i, minibatch in enumerate(dev_set.dev_batches(32)):
                _, _, label_ids, sentence_lengths, _ = minibatch

                labels_ids_predictions, loss = self.predict_batch(minibatch, task_code)
                losses.append(loss)

                for lab, lab_pred, length in zip(label_ids, labels_ids_predictions, sentence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    for (true_t, pred_t) in zip(lab, lab_pred):
                        accs.append(true_t == pred_t)
                        if dev_set.task == 'ner':
                            O_token = self.dm.task_vocabs['ner'].token_to_id['O']
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
                precision = float(precision+1) / (predicted_ner+1)
                recall = (float(recall) / expected_ner)
                f1 = 2*precision*recall/(precision+recall)
                output.update({
                    'expected_ner_count': expected_ner,
                    'predicted_ner_count': predicted_ner,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        if dev_set.task in ('dep'):

            uads = []
            losses = []

            for i, minibatch in enumerate(dev_set.dev_batches(5)):

                word_ids, char_ids, label_ids, sentence_lengths, word_lengths, arcs = minibatch

                fd = {
                    self.word_ids: word_ids,
                    # self.true_labels[task_code]: label_ids,
                    self.sequence_lengths: sentence_lengths,
                    self.dropout: 1,
                    self.word_lengths: word_lengths,
                    self.char_ids: char_ids,
                    self.arc_ids: arcs
                }

                uad, loss = self.sess.run([self.uas, self.dep_loss], feed_dict=fd)
                uads.append(uad)
                losses.append(loss)

            output = {'uad': np.mean(uads), 'loss': np.mean(losses)}

        return output

    def predict_batch(self, minibatch, task_code):

        word_ids, char_ids, label_ids, sentence_lengths, word_lengths = minibatch
        fd = {
            self.word_ids: word_ids,
            self.true_labels[task_code]: label_ids,
            self.sequence_lengths: sentence_lengths,
            self.dropout: 1,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids
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

    def initialize_logger(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        return LoggerInit(
            self.config.setup,
            filename=os.path.join(self.config.log_path, timestamp),
            slack_channel=self.config.slack_channel,
            slack_token=self.config.slack_token
        ).logger
