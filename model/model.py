import tensorflow as tf
import numpy as np
import datetime
import os

from logs.logger_init import LoggerInit

from model.general_model import GeneralModel
from model._model_dep import DEPModel
from model._model_lmo import LMOModel
from model._model_nli import NLIModel
from model._model_sqt import SQTModel


class Model(GeneralModel, DEPModel, LMOModel, NLIModel, SQTModel):

    def task_code(self, task, lang):
        if self.config.crf_sharing:
            return task
        else:
            return task + lang

    def load_embeddings(self, lang):
        emb_path = os.path.join(self.config.emb_path, lang)
        emb_matrix = np.zeros((len(self.dl.lang_vocabs[lang]), self.config.word_emb_size), dtype=np.float)
        with open(emb_path, 'r') as f:
            next(f)
            for line in f:
                try:
                    word, rest = line.split(maxsplit=1)
                except:
                    print(line)
                    raise AttributeError
                if word in self.dl.lang_vocabs[lang]:
                    i = self.dl.lang_vocabs[lang].t2id[word]
                    try:
                        emb_matrix[i] = [float(n) for n in rest.split()]
                    except ValueError:
                        pass # FIXME: sometimes there are two words in embeddings file, but I think it's better to clean the emb files instead
        return emb_matrix

    def _build_graph(self):


        #
        # hyperparameters
        self.learning_rate = tf.placeholder(dtype=self.type, shape=[],
        name="lr")

        self.dropout = tf.placeholder(dtype=self.type, shape=[],
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

        # language flags
        self.language_flags = dict()
        for lang in self.dl.langs():
            self.language_flags[lang] = tf.placeholder_with_default(input=False, shape=[], name="language_flag_%s" % lang)

        #
        # word embeddings
        _word_embeddings = dict()
        for lang in self.dl.langs():
            _word_embeddings[lang] = tf.get_variable(
                dtype=self.type,
                # shape=[None, self.config.word_emb_size],
                initializer=tf.cast(self.load_embeddings(lang), self.type),
                trainable=self.config.train_emb,
                name="word_embeddings_%s" % lang)

        lambdas = lambda lang: lambda: _word_embeddings[lang]
        cases = dict([(self.language_flags[lang], lambdas(lang)) for lang in self.dl.langs()]) # FIXME: tf.merge?
        cased_word_embeddings = tf.case(cases) # TODO: when all are false it will pick default
        self.word_embeddings = tf.nn.embedding_lookup(cased_word_embeddings, self.word_ids,
        name="word_embeddings_lookup")

        if self.config.char_level:
            _char_embeddings = tf.get_variable(
                dtype=self.type,
                shape=(len(self.dl.char_vocab), self.config.char_emb_size),
                name="char_embeddings"
            )
            self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
            name="char_embeddings_lookup")
            self.sh = tf.shape(self.char_embeddings)

            with tf.variable_scope("char_bi-lstm"):

                max_sentence_length = tf.shape(self.char_embeddings)[1]
                max_word_length = tf.shape(self.char_embeddings)[2]
                # TODO: Check if it is really sensitive to padding
                self.char_embeddings = tf.reshape(self.char_embeddings, [-1, max_word_length, self.config.char_emb_size])
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.char_lstm_size)
                word_lengths = tf.reshape(self.word_lengths, [-1])
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.char_embeddings,
                    sequence_length=word_lengths, dtype=self.type)
                # shape(batch_size*max_sentence, max_word, 2 x word_lstm_size)
                char_lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                char_lstm_output = tf.reduce_sum(char_lstm_output, 1)

                word_lengths = tf.cast(word_lengths, dtype=self.type)
                word_lengths = tf.add(word_lengths, 1e-8)
                word_lengths = tf.expand_dims(word_lengths, 1)
                word_lengths = tf.tile(word_lengths, [1, 2 * self.config.char_lstm_size])

                char_lstm_output = tf.divide(char_lstm_output, word_lengths)
                char_lstm_output = tf.reshape(char_lstm_output, (-1, max_sentence_length, 2*self.config.char_lstm_size))
                self.word_embeddings = tf.concat([self.word_embeddings, char_lstm_output], axis=-1)

        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)

        #
        # bi-lstm
        with tf.variable_scope("word_bi-lstm"):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.word_lstm_size)
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.word_lstm_size)
            (self.lstm_fw, self.lstm_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=self.type)
            # shape(batch_size, max_length, 2 x word_lstm_size)
            self.lstm_fw = tf.nn.dropout(self.lstm_fw, self.dropout)
            self.lstm_bw = tf.nn.dropout(self.lstm_bw, self.dropout)
            self.word_lstm_output = tf.concat([self.lstm_fw, self.lstm_bw], axis=-1)

        self.true_labels = dict()
        self.predicted_labels = dict()
        self.trans_params = dict()
        self.loss = dict()
        self.train_op = dict()
        self.gradient_norm = dict()

        used_task_codes = []
        for (task, lang) in self.config.tasks:
            task_code = self.task_code(task, lang)
            if task_code not in used_task_codes:
                if task in ('ner', 'pos'):
                    self.add_crf(task, task_code)
                if task in ('dep'):
                    self.add_dep(task_code)
                if task in ('nli'):
                    self.add_nli(task_code)
                used_task_codes.append(task_code)
            if task in ('lmo'):
                self.add_lmo(task_code, lang)



    def run_experiment(self, train, test, epochs):
        self.logger.log_critical('%s: Run started.' % self.config.server_name)
        self.name = ' '.join([' '.join(t) for t in train])
        self.logger.log_message("Now training " + self.name)
        start_time = datetime.datetime.now()
        self.logger.log_message(start_time)
        #self.logger.log_message(self.config.dump()) # FIXME
        self.epoch = 1
        for i in range(epochs):
            self.run_epoch(train=train, test=test)
            if i == 0:
                epoch_time = datetime.datetime.now() - start_time
                self.logger.log_critical('%s ETA: %s' % (self.config.server_name, str(start_time + epoch_time * self.config.epochs)))
        if self.config.save_parameters:
            self.saver.save(self.sess, os.path.join(self.config.model_path, self.timestamp+".model"))
        end_time = datetime.datetime.now()
        self.logger.log_message(end_time)
        self.logger.log_message('Training took: '+str(end_time-start_time))
        self.logger.log_critical('%s: Run done.' % self.config.server_name)

    def run_epoch(self, train, test):


        # FIXME: how to choose proper generator?
        train_sets = [(dt, dt.train_file_generator(self.config.batch_size)) for dt in self.dl.find(role='train')] # FIXME: train sa vytvara zakazdym nanovo
        dev_sets = [(dt, dt.test_file_generator(self.config.batch_size, limit=1000)) for dt in self.dl.find(role='train')]
        dev_sets += [(dt, dt.test_file_generator(self.config.batch_size)) for dt in self.dl.find(role='dev')]
        dev_sets += [(dt, dt.test_file_generator(self.config.batch_size)) for dt in self.dl.find(role='test')]

        # dev_sets = [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in train]
        # dev_sets += [self.dm.fetch_dataset(task, lang, 'train-dev') for (task, lang) in train]
        # dev_sets += [self.dm.fetch_dataset(task, lang, 'test') for (task, lang) in train]
        # dev_sets += [self.dm.fetch_dataset(task, lang, 'dev') for (task, lang) in test]

        # if self.config.train_only: # TODO: odstranit po experimente (aj v hparams)
        #     task, lang = self.config.train_only.split('-')
        #     train_sets = [self.dm.fetch_dataset(task, lang, 'train')]
        #     dev_sets = [self.dm.fetch_dataset(task, lang, 'train-dev'),
        #                 self.dm.fetch_dataset(task, lang, 'test'),
        #                 self.dm.fetch_dataset(task, lang, 'dev')]

        for _ in range(self.config.epoch_steps):
            for st, ite in train_sets:
                task_code = self.task_code(st.task, st.lang)

                if st.task in ('ner', 'pos'):

                    SQTModel.train(self, next(ite), task_code)

                if st.task in ('dep'):

                    DEPModel.train(self, next(ite), task_code)

                if st.task in ('nli'):

                    NLIModel.train(self, next(ite), task_code)

                if st.task in ('lmo'):

                    LMOModel.train(self, next(ite), task_code)


        self.logger.log_message("End of epoch " + str(self.epoch))

        for st, ite in dev_sets:
            metrics = {
                'language': st.lang,
                'task': st.task,
                'role': st.role,
                'epoch': self.epoch,
                'run': self.name
            }
            metrics.update(self.run_evaluate(st, ite))
            self.logger.log_result(metrics)

        self.epoch += 1

    def run_evaluate(self, dev_set, set_iterator):
        task_code = self.task_code(dev_set.task, dev_set.lang)

        if dev_set.task in ('ner', 'pos'):

            return SQTModel.evaluate(self, set_iterator, task_code)

        if dev_set.task in ('dep'):

            return DEPModel.evaluate(self, set_iterator, task_code)

        if dev_set.task in ('nli'):

            return NLIModel.evaluate(self, set_iterator, task_code)

        if dev_set.task in ('lmo'):

            return LMOModel.evaluate(self, set_iterator, task_code)



