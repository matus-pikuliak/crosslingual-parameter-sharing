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

    def add_crf(self, task, task_code):
        with tf.variable_scope(task_code):
            tag_count = len(self.dm.task_vocabs[task])

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
            grads, vs = zip(*self.optimizer.compute_gradients(self.loss[task_code]))
            self.gradient_norm[task_code] = tf.global_norm(grads)
            if self.config.clip > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
            self.train_op[task_code] = self.optimizer.apply_gradients(zip(grads, vs))

    def build_graph(self):

        self.true_labels = dict()
        self._true_labels = dict()
        self.predicted_labels = dict()
        self.trans_params = dict()
        self.loss = dict()
        self.train_op = dict()
        self.gradient_norm = dict()
        self.dataset = dict()
        self.iterator = dict()

        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
        name="lr")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
        name="dropout")

        #
        # inputs
        # shape = (batch size, max_sentence_length)
        # TODO: Treba zjednotit tieto nazvy s nazvami v module dataset
        self._word_ids = tf.placeholder(tf.int32, shape=[None, None],
        name="word_ids")
        # shape = (batch size)
        self._sequence_lengths = tf.placeholder(tf.int32, shape=[None],
        name="sequence_lengths")
        # shape = (batch_size, max_sentence_length, max_word_length)
        self._char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
        name="char_ids")
        # shape = (batch_size, max_sentence_length)
        self._word_lengths = tf.placeholder(tf.int32, shape=[None, None],
        name="word_lengths")

        task_codes = set([self.task_code(task, lang) for (task, lang) in self.dm.tls])

        for task_code in task_codes:
            # expected output
            # shape = (batch%size, max_length)
            self._true_labels[task_code] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")

            self.dataset[task_code] = tf.data.Dataset.from_tensor_slices((
                self._word_ids,
                self._char_ids,
                self._true_labels[task_code],
                self._sequence_lengths,
                self._word_lengths,
            )).batch(4)
            self.iterator[task_code] = self.dataset[task_code].make_initializable_iterator()
            self.word_ids, self.char_ids, self.true_labels[task_code], self.sequence_lengths, self.word_lengths = \
            self.iterator[task_code].get_next()

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
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.char_embeddings,
                    sequence_length=self.word_lengths_seq, dtype=tf.float32)
                # shape(batch_size*max_sentence, max_word, 2 x word_lstm_size)
                self.char_lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                self.char_lstm_output = tf.reduce_mean(self.char_lstm_output, 1)
                self.char_lstm_output = tf.reshape(self.char_lstm_output, (-1, max_sentence_length, 2*self.config.char_lstm_size))
                self.word_embeddings = tf.concat([self.word_embeddings, self.char_lstm_output], axis=-1, name="pomoc")

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

        used_task_codes = []
        for (task, lang) in self.dm.tls:
            task_code = self.task_code(task, lang)
            if task_code not in used_task_codes:
                self.add_crf(task, task_code)
                used_task_codes.append(task_code)

        if self.config.use_gpu:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1}))
        self.sess.run(tf.global_variables_initializer())

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
        self.logger.log_message('Training took:'+str(end_time-start_time))
        self.logger.log_critical('Run done.')

    def run_epoch(self, epoch_id,
                  train,
                  test):

        train_sets = [self.dm.fetch_dataset(task, lang, 'train') for (task, lang) in train]
        dev_sets = [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in train]
        dev_sets += [self.dm.fetch_dataset(task, lang, 'train-dev') for (task, lang) in train]
        dev_sets += [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in test]

        for st in train_sets:
            task_code = self.task_code(st.task, st.lang)
            fd = st.final_samples()
            self.sess.run(self.iterator[task_code].initializer, feed_dict={
                self._word_ids: fd[0],
                self._char_ids: fd[1],
                self._true_labels[task_code]: fd[2],
                self._sequence_lengths: fd[3],
                self._word_lengths: fd[4],
            })
            for i in range(5):
                _, train_loss, gradient_norm = self.sess.run(
                    [self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]],
                    feed_dict = {self.learning_rate: self.config.learning_rate,
                          self.dropout: self.config.dropout}
                )

                # word_ids, char_ids, label_ids, sentence_lengths, word_lengths = minibatch
                #
                # fd = {
                #     self.word_ids: word_ids,
                #     self.true_labels[task_code]: label_ids,
                #     self.sequence_lengths: sentence_lengths,
                #     self.word_lengths: word_lengths,
                #     self.char_ids: char_ids,
                #     self.learning_rate: self.config.learning_rate,
                #     self.dropout: self.config.dropout
                # }
                #
                # _, train_loss, gradient_norm = self.sess.run(
                #     [self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]]
                #     , feed_dict=fd
                # )

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
        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0

        st = dev_set
        task_code = self.task_code(st.task, st.lang)
        fd = st.final_samples()
        self.sess.run(self.iterator[task_code].initializer, feed_dict={
            self._word_ids: fd[0],
            self._char_ids: fd[1],
            self._true_labels[task_code]: fd[2],
            self._sequence_lengths: fd[3],
            self._word_lengths: fd[4],
        })
        for i in range(10):
            label_ids, sentence_lengths, labels_ids_predictions, loss = self.predict_batch(task_code)
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
        return output

    def predict_batch(self, task_code):

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        label_ids, sentence_lengths, logits, trans_params, loss = self.sess.run(
            [self.true_labels, self.sequence_lengths, self.predicted_labels[task_code], self.trans_params[task_code], self.loss[task_code]],
            feed_dict={self.dropout: 1}
        )

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sentence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return label_ids, sentence_lengths, viterbi_sequences, loss

    def initialize_logger(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        return LoggerInit(
            self.config.setup,
            filename=os.path.join(self.config.log_path, timestamp),
            slack_channel=self.config.slack_channel,
            slack_token=self.config.slack_token
        ).logger
