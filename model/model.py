import tensorflow as tf
import numpy as np
import json
from data import dataset

class Model:

    def __init__(self, cache, config):
        self.config = config
        self.cache = cache
        self.sess = None

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def build_graph(self):

        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
        name="lr")

        #
        # inputs
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
        name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
        name="sequence_lengths")

        # optimizer
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        #
        # word embeddings
        _word_embeddings = tf.get_variable(
            dtype=tf.float32,
            #shape=[None, self.config.word_emb_size],
            initializer=tf.cast(self.cache.embeddings, tf.float32),
            trainable=self.config.train_embeddings,
            name="word_embeddings")
        self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
        name="word_embeddings_lookup")
        #self.word_embeddings = tf.nn.dropout(self.word_embeddings, 0.5)

        #
        # bi-lstm
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            # shape(batch_size, max_length, 2 x lstm_size)
            lstm_output = tf.concat([output_fw, output_bw], axis=-1)
            #lstm_output = tf.nn.dropout(lstm_output, 0.5)

        #
        # ner
        if "ner" in self.cache.task_dicts:
            with tf.variable_scope("ner"):
                ner_tag_count = len(self.cache.task_dicts['ner'][0])

                # expected output
                # shape = (batch%size, max_length)
                self.ner_true_labels = tf.placeholder(tf.int32, shape=[None, None],
                name="labels")
                # projection
                W = tf.get_variable(dtype=tf.float32, shape=[2 * self.config.lstm_size, ner_tag_count],
                name="proj_weights")
                b = tf.get_variable(dtype=tf.float32, shape=[ner_tag_count], initializer=tf.zeros_initializer(),
                name="proj_biases")

                max_length = tf.shape(lstm_output)[1] # TODO: toto moze byt vstupom z vonku?
                reshaped_output = tf.reshape(lstm_output, [-1, 2 * self.config.lstm_size]) # We can apply the weight on all outputs of LSTM now
                ner_proj = tf.matmul(reshaped_output, W) + b
                self.ner_predicted_labels = tf.reshape(ner_proj, [-1, max_length, ner_tag_count])

                # loss
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.ner_predicted_labels,
                    self.ner_true_labels,
                    self.sequence_lengths)
                self.ner_trans_params = trans_params  # need to evaluate it for decoding
                self.ner_loss = tf.reduce_mean(-log_likelihood)

                # training
                if self.config.clip > 0:  # gradient clipping if clip is positive
                    grads, vs = zip(*optimizer.compute_gradients(self.ner_loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
                    self.ner_train_op = optimizer.apply_gradients(zip(grads, vs))
                else:
                    self.ner_train_op = optimizer.minimize(self.ner_loss)

        #
        # pos
        # the same code as ner part, if there are more sequence labeling tasks it would be worth it to abstract it into
        # its own function perhaps.
        if "pos" in self.cache.task_dicts:
            with tf.variable_scope("pos"):
                pos_tag_count = len(self.cache.task_dicts['pos'][0])

                # expected output
                # shape = (batch%size, max_length)
                self.pos_true_labels = tf.placeholder(tf.int32, shape=[None, None],
                name="labels")
                # projection
                W = tf.get_variable(dtype=tf.float32, shape=[2 * self.config.lstm_size, pos_tag_count],
                name="proj_weights")
                b = tf.get_variable(dtype=tf.float32, shape=[pos_tag_count], initializer=tf.zeros_initializer(),
                name="proj_biases")

                max_length = tf.shape(lstm_output)[1] # TODO: toto moze byt vstupom z vonku?
                reshaped_output = tf.reshape(lstm_output, [-1, 2 * self.config.lstm_size]) # We can apply the weight on all outputs of LSTM now
                pos_proj = tf.matmul(reshaped_output, W) + b
                self.pos_predicted_labels = tf.reshape(pos_proj, [-1, max_length, pos_tag_count])

                # loss
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.pos_predicted_labels,
                    self.pos_true_labels,
                    self.sequence_lengths)
                self.pos_trans_params = trans_params  # need to evaluate it for decoding
                self.pos_loss = tf.reduce_mean(-log_likelihood)

                # training
                if self.config.clip > 0:  # gradient clipping if clip is positive
                    grads, vs = zip(*optimizer.compute_gradients(self.pos_loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
                    self.pos_train_op = optimizer.apply_gradients(zip(grads, vs))
                else:
                    self.pos_train_op = optimizer.minimize(self.pos_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) # TODO: check what is default initializer

    def run_epoch(self, epoch_id,
                  train,
                  test,
                  learning_rate=None):

        train_sets = [self.cache.fetch_dataset(task, lang, 'train') for (task, lang) in train]
        dev_sets = [self.cache.fetch_dataset(task, lang, 'dev') for (task, lang) in train]
        dev_sets += [self.cache.fetch_dataset(task, lang, 'dev') for (task, lang) in test]

        for _ in xrange(self.config.epoch_steps):
            for train_set in train_sets:
                minibatch = train_set.next_batch(self.config.batch_size)

                if train_set.task == 'ner':
                    (word_ids, labels, lengths) = minibatch
                    fd = {
                        self.word_ids: word_ids,
                        self.ner_true_labels: labels,
                        self.sequence_lengths: lengths,
                        self.learning_rate: (learning_rate or self.config.learning_rate)
                    }
                    _, train_loss = self.sess.run([self.ner_train_op, self.ner_loss], feed_dict=fd)

                if train_set.task == 'pos':
                    (word_ids, labels, lengths) = minibatch
                    fd = {
                        self.word_ids: word_ids,
                        self.pos_true_labels: labels,
                        self.sequence_lengths: lengths,
                        self.learning_rate: (learning_rate or self.config.learning_rate)
                    }
                    _, train_loss = self.sess.run([self.pos_train_op, self.pos_loss], feed_dict=fd)

        with open('./logs.txt', 'a') as f:
            f.write('end of epoch '+str(epoch_id+1)+'\n')

        for sample_set in dev_sets + train_sets:
            if sample_set.role == 'train':
               sample_set = dataset.Dataset(sample_set.language, sample_set.task, "train-stats",
                                                sample_set.samples[:1000])
            metrics = {'language': sample_set.language, 'task': sample_set.task, 'role': sample_set.role}
            metrics.update(self.run_evaluate(sample_set))
            with open('./logs.txt', 'a') as f:
                f.write(' '.join(['%s: %s' % (k, metrics[k]) for k in metrics])+'\n')

    def run_evaluate(self, dev_set):
        accs = []
        losses = []
        expected_ner = 0
        predicted_ner = 0
        precision = 0
        recall = 0
        O_token = self.cache.task_dicts['ner'][1]['O']

        for i, (words, labels, lengths) in enumerate(dev_set.dev_batches(self.config.batch_size)):
            labels_predictions, loss = self.predict_batch(words, labels, lengths, dev_set.task)
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
            output.update({
                'expected_ner_count': expected_ner,
                'predicted_ner_count': predicted_ner,
                'precision': float(precision) / (predicted_ner+0.001),
                'recall': float(recall) / (expected_ner+0.001)
            })
        return output

    def predict_batch(self, words, labels, lengths, task):

        fd = {
            self.word_ids: words,
            self.sequence_lengths: lengths
        }

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        if task == "ner":
            fd[self.ner_true_labels] = labels
            logits, trans_params, loss = self.sess.run(
                [self.ner_predicted_labels, self.ner_trans_params, self.ner_loss], feed_dict=fd)
        if task == "pos":
            fd[self.pos_true_labels] = labels
            logits, trans_params, loss = self.sess.run(
                [self.pos_predicted_labels, self.pos_trans_params, self.pos_loss], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, loss
