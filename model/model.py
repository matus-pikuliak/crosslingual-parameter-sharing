import tensorflow as tf
import numpy as np
from data import dataset
import datetime


class Model:

    def __init__(self, cache, config, logger):
        self.config = config
        self.cache = cache
        self.sess = None
        self.logger = logger

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
            tag_count = len(self.cache.task_dicts[task][0])

            # expected output
            # shape = (batch%size, max_length)
            self.true_labels[task_code] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")
            # projection
            W = tf.get_variable(dtype=tf.float32, shape=[2 * self.config.lstm_size, tag_count],
                                name="proj_weights")
            b = tf.get_variable(dtype=tf.float32, shape=[tag_count], initializer=tf.zeros_initializer(),
                                name="proj_biases")

            max_length = tf.shape(self.lstm_output)[1]  # TODO: toto moze byt vstupom z vonku?
            reshaped_output = tf.reshape(self.lstm_output, [-1,
                                                       2 * self.config.lstm_size])  # We can apply the weight on all outputs of LSTM now
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

        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
        name="lr")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
        name="dropout")

        #
        # inputs
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
        name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
        name="sequence_lengths")

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
            initializer=tf.cast(self.cache.embeddings, tf.float32),
            trainable=self.config.train_embeddings,
            name="word_embeddings")
        self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
        name="word_embeddings_lookup")
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)

        #
        # bi-lstm
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            # shape(batch_size, max_length, 2 x lstm_size)
            self.lstm_output = tf.concat([output_fw, output_bw], axis=-1)
            self.lstm_output = tf.nn.dropout(self.lstm_output, self.dropout)

        self.true_labels = dict()
        self.predicted_labels = dict()
        self.trans_params = dict()
        self.loss = dict()
        self.train_op = dict()
        self.gradient_norm = dict()

        used_task_codes = []
        for (task, lang) in self.cache.task_langs:
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
        self.name = ' '.join([' '.join(t) for t in train])
        self.logger.log_m("Now training " + self.name)
        start_time = datetime.datetime.now()
        self.logger.log_m(str(start_time))
        self.logger.log_m(self.config.dump())
        for i in xrange(epochs):
            self.run_epoch(i, train=train, test=test)
        end_time = datetime.datetime.now()
        self.logger.log_m(str(end_time))
        self.logger.log_m('Training took:'+str(end_time-start_time))

    def run_epoch(self, epoch_id,
                  train,
                  test,
                  learning_rate=None):

        train_sets = [self.cache.fetch_dataset(task, lang, 'train') for (task, lang) in train]
        dev_sets = [self.cache.fetch_dataset(task, lang, 'dev') for (task, lang) in train]
        dev_sets += [self.cache.fetch_dataset(task, lang, 'dev') for (task, lang) in test]

        for train_set in train_sets:
            train_set.eval_set = dataset.Dataset(train_set.language, train_set.task, "train-1k",
                                                train_set.samples[:1000])

        for _ in xrange(self.config.epoch_steps):
            for train_set in train_sets:
                task_code = self.task_code(train_set.task, train_set.language)
                minibatch = train_set.next_batch(self.config.batch_size)
                (word_ids, labels, lengths) = minibatch
                fd = {
                    self.word_ids: word_ids,
                    self.true_labels[task_code]: labels,
                    self.sequence_lengths: lengths,
                    self.learning_rate: (learning_rate or self.config.learning_rate),
                    self.dropout: self.config.dropout
                }
                _, train_loss, gradient_norm = self.sess.run([self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]], feed_dict=fd)

        self.logger.log_m("End of epoch " + str(epoch_id+1))

        for sample_set in dev_sets + train_sets:
            if sample_set.role == 'train':
               sample_set = sample_set.eval_set
            metrics = {
                'language': sample_set.language,
                'task': sample_set.task,
                'role': sample_set.role,
                'epoch': epoch_id + 1,
                'run': self.name
            }
            metrics.update(self.run_evaluate(sample_set))
            self.logger.log_r(metrics)

    def run_evaluate(self, dev_set):
        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0
        O_token = self.cache.task_dicts['ner'][1]['O']
        task_code = self.task_code(dev_set.task, dev_set.language)

        for i, (words, labels, lengths) in enumerate(dev_set.dev_batches(512)): # TODO: toto netreba robit v malych batchoch
            labels_predictions, loss = self.predict_batch(words, labels, lengths, task_code)
            losses.append(loss)

            for lab, lab_pred, length in zip(labels, labels_predictions, lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                for (true_t, pred_t) in zip(lab, lab_pred):
                    accs.append(true_t == pred_t)
                    if dev_set.task == 'ner':
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

    def predict_batch(self, words, labels, lengths, task_code):

        fd = {
            self.word_ids: words,
            self.sequence_lengths: lengths,
            self.true_labels[task_code]: labels,
            self.dropout: 1
        }

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params, loss = self.sess.run(
            [self.predicted_labels[task_code], self.trans_params[task_code], self.loss[task_code]], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, loss
