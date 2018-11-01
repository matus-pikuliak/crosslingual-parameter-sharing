import tensorflow as tf
import numpy as np
import datetime
import os

from logs.logger_init import LoggerInit
from model._layer_ner import NERLayer
from model._layer_pos import POSLayer
from model.dataset_iterator import DatasetIterator
from model.general_model import GeneralModel

# FIXME: funguje vsetky dataset size limitations? aj train_only?
from model.layer import Layer

class Model(GeneralModel):

    def __init__(self, *args, **kwargs):
        GeneralModel.__init__(self, *args, **kwargs)
        self.langs = self.dl.langs()
        self.tasks = self.dl.tasks()

    def task_code(self, task, lang):
        if task == 'lmo':
            return (task, lang)
        if self.config.task_layer_sharing:
            return task
        else:
            return (task, lang)

    def _build_graph(self):
        self.add_inputs()
        word_repr = self.add_word_processing()
        cont_repr = self.add_sentence_processing(word_repr)
        self.add_task_layers(cont_repr)

    def add_inputs(self):
        # shape = (batch size, max_sentence_length)
        self.word_ids = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="word_ids"
        )

        # shape = (batch size)
        self.sentence_length = tf.placeholder(
            dtype=tf.int32, shape=[None], name='sentence_length'
        )

        # shape = (batch_size, max_sentence_length, max_word_length)
        self.char_ids = tf.placeholder(
            dtype=tf.int32, shape=[None, None, None], name='char_ids'
        )

        # shape = (batch_size, max_sentence_length)
        self.word_lengths = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='word_lengths'
        )

        self.lang_flags = {
            lang: tf.placeholder_with_default(
                input=False, shape=[], name=f'language_flag_{lang}'
            )
            for lang in self.langs
        }

    def add_word_processing(self):

        word_emb_matrix = self.add_word_emb_matrices()
        self.word_embeddings = tf.nn.embedding_lookup(
            params=word_emb_matrix, ids=self.word_ids, name="word_embeddings_lookup"
        )

        if self.config.char_level:
            char_embeddings = self.add_char_embeddings()
            self.word_embeddings = tf.concat([self.word_embeddings, char_embeddings], axis=-1)

        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)
        self.word_embeddings = tf.where(tf.is_nan(self.word_embeddings), tf.zeros_like(self.word_embeddings), self.word_embeddings)
        # FIXME: why do NaNs go into gradient? they should not go into word level lstm at all
        # TODO: Are zero length words time consuming?
        return self.word_embeddings

    def add_word_emb_matrices(self):
        emb_matrices = {
            lang: tf.get_variable(
                dtype=self.type,
                initializer=tf.cast(self.load_embeddings(lang), self.type),
                trainable=self.config.train_emb,
                name=f'word_embedding_matrix_{lang}')
            for lang in self.langs
        }

        pred_fn_pairs = {
            self.lang_flags[lang]: (lambda lang: lambda: emb_matrices[lang])(lang)
            for lang in self.langs
        }
        return tf.case(pred_fn_pairs=pred_fn_pairs, exclusive=True)  # TODO: when all are false it will pick default?

    def add_char_embeddings(self):
        emb_matrix = tf.get_variable(
            dtype=self.type,
            shape=(len(self.dl.char_vocab), self.config.char_emb_size),
            name="char_embeddings")

        char_embeddings = tf.nn.embedding_lookup(
            params=emb_matrix,
            ids=self.char_ids,
            name="char_embeddings_lookup")

        shape = tf.shape(char_embeddings)
        max_sentence, max_word = shape[1], shape[2]

        char_embeddings = tf.reshape(char_embeddings, [-1, max_word, self.config.char_emb_size])
        word_lengths = tf.reshape(self.word_lengths, [-1])
        char_lstm_out = self.lstm(
            inputs=char_embeddings,
            sequence_lengths=word_lengths,
            cell_size=self.config.char_lstm_size,
            name_scope='char_bilstm',
            avg_pool=True,
            dropout=False)

        char_lstm_out = tf.reshape(char_lstm_out, [-1, max_sentence, 2 * self.config.char_lstm_size])
        return char_lstm_out

    def add_sentence_processing(self, word_repr):
        return self.lstm(
            inputs=word_repr,
            sequence_lengths=self.sentence_length,
            cell_size=self.config.word_lstm_size,
            name_scope='word_bilstm')

    def add_task_layers(self, cont_repr):
        for (task, lang) in self.config.tasks:
            task_code = self.task_code(task, lang)
            if task_code not in Layer.layers:
                layer_class = globals()[f'{task.upper()}Layer']
                new_layer = layer_class(self, task, lang, cont_repr)
                Layer.layers[task_code] = new_layer

    def run_experiment(self, train, test, epochs):
        self.logger.log_critical('%s: Run started.' % self.config.server_name)
        self.name = ' '.join([' '.join(t) for t in train])
        self.logger.log_message("Now training " + self.name)
        start_time = datetime.datetime.now()
        self.logger.log_message(start_time)
        #self.logger.log_message(self.config.dump()) # FIXME
        self.epoch = 1
        for i in range(epochs):
            self.run_epoch()
            if i == 0:
                epoch_time = datetime.datetime.now() - start_time
                self.logger.log_critical('%s ETA: %s' % (self.config.server_name, str(start_time + epoch_time * self.config.epochs)))
        if self.config.save_parameters:
            self.saver.save(self.sess, os.path.join(self.config.model_path, self.timestamp+".model"))
        end_time = datetime.datetime.now()
        self.logger.log_message(end_time)
        self.logger.log_message('Training took: '+str(end_time-start_time))
        self.logger.log_critical('%s: Run done.' % self.config.server_name)

    def run_epoch(self):

        train_sets = [DatasetIterator(dt, self.config, self.task_code) for dt in self.dl.find(role='train')]
        eval_sets = [DatasetIterator(dt, self.config, self.task_code, is_train=False) for dt in self.dl.datasets]

        # FIXME: Dopln train_only funkcionalitu (je to len redukcia train_sets?)

        for _ in range(self.config.epoch_steps):
            for st in train_sets:
                st.layer.train(next(st.iterator), st.dataset)

        self.logger.log_message(f'Epoch {self.epoch} training done.')

        for st in eval_sets:
            results = st.layer.evaluate(st.iterator, st.dataset)
            results.update({
                'language': st.dataset.lang,
                'task': st.dataset.task,
                'role': st.dataset.role,
                'epoch': self.epoch
            })
            self.logger.log_result(results)

        self.epoch += 1

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

    def lstm(self, inputs, sequence_lengths, cell_size, name_scope, avg_pool=False, dropout=True):
        with tf.variable_scope(name_scope):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(cell_size)
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(cell_size)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_lengths, dtype=self.type)

            # shape = (batch_size, max(sequence_lengths), 2*cell_size)
            out = tf.concat([out_fw, out_bw], axis=-1)

            if avg_pool:
                out = tf.reduce_sum(out, axis=1)
                div_mask = tf.cast(sequence_lengths, tf.float32)
                div_mask = tf.expand_dims(div_mask, axis=-1)  # Axis needed for proper dimension broadcasting.
                out = tf.divide(out, div_mask)

            if dropout:
                out = tf.nn.dropout(out, self.dropout)

            return out
