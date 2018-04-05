import tensorflow as tf
import numpy as np
from utils import general_utils as gu

class Model:

    def __init__(self, cache, config):
        self.config = config
        self.cache = cache
        self.sess = None

    def build_graph(self):

        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
        name="lr")
        self.clipping_threshold = tf.placeholder(dtype=tf.float32, shape=[],
        name="clip")

        #
        # inputs
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
        name="word_ids")
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
        name="sequence_lengths")

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

        #
        # ner
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
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            if self.clipping_threshold > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.ner_loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.clipping_threshold)
                self.ner_train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.ner_train_op = optimizer.minimize(self.ner_loss)

        #
        # pos
        #

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) # TODO: check what is default initializer

    def run_epoch(self,
                  train,
                  dev,
                  learning_rate=None,
                  clipping_threshold=None):


        minibatches = train.minibatches(self.config.batch_size)
        for _ in xrange(100):

            for i, (words, labels, lengths) in enumerate(minibatches):

                fd = {
                    self.word_ids: words,
                    self.ner_true_labels: labels,
                    self.sequence_lengths: lengths,
                    self.learning_rate: (learning_rate or self.config.learning_rate),
                    self.clipping_threshold: (clipping_threshold or self.config.clip)
                }

                _, train_loss = self.sess.run([self.ner_train_op, self.ner_loss], feed_dict=fd)

                print train_loss

        #metrics = self.run_evaluate(dev)
        #print "dev acc: "+metrics["acc"]

    def run_evaluate(self, test):
        accs = []
        for i, (words, labels) in enumerate(test.minibatches(self.config.batch_size)):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

        acc = np.mean(accs)

        return {"acc": 100 * acc}

    def predict_batch(self, words):

        fd, sequence_lengths = self.get_feed_dict(words)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
            [self.ner_proj, self.ner_trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths
