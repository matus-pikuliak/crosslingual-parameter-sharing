import itertools
import random
import types

import numpy as np
import tensorflow as tf

from data.embedding import Embeddings
from model.dataset_iterator import DatasetIterator

from constants import LOG_CRITICAL, LOG_MESSAGE, LOG_RESULT

class Model:

    def __init__(self, task, lang, orchestrator, data_loader, config, logger):
        self.task = task
        self.lang = lang
        self.orch = orchestrator
        self.dl = data_loader
        self.config = config
        self.logger = logger

        self.n = types.SimpleNamespace()

        # TODO: add iterators


    def build_graph(self):
        self.add_inputs()
        self.add_utils()
        self.add_word_processing()
        self.add_sentence_processing()
        self.add_task_layer()

    def add_inputs(self):
        # shape = (batch size, max_sentence_length)
        self.n.word_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="word_ids")

        # shape = (batch size)
        self.n.sentence_lengths = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='sentence_lengths')

        # shape = (batch_size, max_sentence_length, max_word_length)
        self.n.char_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None, None],
            name='char_ids')

        # shape = (batch_size, max_sentence_length)
        self.n.word_lengths = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='word_lengths')

        self.n.lang_id = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name='lang_id')

        self.n.task_id = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name='task_id')

    def add_utils(self):
        """
        Several useful nodes.
        """
        self.n.sentence_lengths_mask = tf.sequence_mask(self.n.sentence_lengths)
        self.n.total_batch_length = tf.reduce_sum(self.n.sentence_lengths)
        self.n.batch_size = tf.shape(self.n.word_ids)[0]
        self.n.max_length = tf.shape(self.n.word_ids)[1]

    def add_word_processing(self):

        word_emb_matrix = self.add_word_emb_matrices()
        word_embeddings = tf.nn.embedding_lookup(
            params=word_emb_matrix,
            ids=self.word_ids,
            name="word_embeddings_lookup")

        if self.config.char_level:
            char_embeddings = self.add_char_embeddings()
            word_embeddings = tf.concat(
                values=[word_embeddings, char_embeddings],
                axis=-1)

        word_embeddings = tf.nn.dropout(
            x=word_embeddings,
            keep_prob=self.dropout)

        return word_embeddings

    def add_word_emb_matrices(self):
        emb_matrices = {
            lang: tf.get_variable(
                dtype=tf.float32,
                initializer=tf.cast(self.load_embeddings(lang), tf.float32),
                trainable=self.config.train_emb,
                name=f'word_embedding_matrix_{lang}')
            for lang in self.langs
        }

        pred_fn_pairs = {
            self.lang_flags[lang]: (lambda lang: lambda: emb_matrices[lang])(lang)
            for lang in self.langs
        }
        return tf.case(
            pred_fn_pairs=pred_fn_pairs,
            exclusive=True)

    def load_embeddings(self, lang):
        emb = Embeddings(lang, self.config)
        return emb.matrix(self.dl.lang_vocabs[lang])

    def add_char_embeddings(self):
        emb_matrix = tf.get_variable(
            dtype=tf.float32,
            shape=(len(self.dl.char_vocab), self.config.char_emb_size),
            name="char_embeddings")

        char_embeddings = tf.nn.embedding_lookup(
            params=emb_matrix,
            ids=self.char_ids,
            name="char_embeddings_lookup")

        shape = tf.shape(char_embeddings)
        max_sentence, max_word = shape[1], shape[2]

        char_embeddings = tf.reshape(
            tensor=char_embeddings,
            shape=[-1, max_word, self.config.char_emb_size])
        word_lengths = tf.reshape(
            tensor=self.word_lengths,
            shape=[-1])
        char_lstm_out = self.lstm(
            inputs=char_embeddings,
            sequence_lengths=word_lengths,
            cell_size=self.config.char_lstm_size,
            name_scope='char_bilstm',
            avg_pool=True,
            dropout=False)

        char_lstm_out = tf.reshape(
            tensor=char_lstm_out,
            shape=[-1, max_sentence, 2 * self.config.char_lstm_size])
        char_lstm_out = tf.where(
            condition=tf.is_nan(char_lstm_out),
            x=tf.zeros_like(char_lstm_out),
            y=char_lstm_out)

        return char_lstm_out

    def add_sentence_processing(self, word_repr, task, lang):

        lstms = []

        def get_lstm(name_scope):
            return self.lstm(
                inputs=word_repr,
                sequence_lengths=self.sentence_lengths,
                cell_size=self.config.word_lstm_size,
                name_scope=name_scope)

        if self.config.word_lstm_private:
            lstms.append(get_lstm(f'word_bilstm_{task}_{lang}'))

        if self.config.word_lstm_task:
            lstms.append(get_lstm(f'word_bilstm_{task}'))

        if self.config.word_lstm_lang:
            lstms.append(get_lstm(f'word_bilstm_{lang}'))

        if self.config.word_lstm_global:
            lstms.append(get_lstm(f'word_bilstm'))

        assert(len(lstms) > 0)

        self.frobenius = self.add_frobenius(*(
            tf.boolean_mask(
                tensor=lstm,
                mask=self.sentence_lengths_mask)
            for lstm
            in lstms))

        if len(lstms) == 1:
            return lstms[0]

        fw, bw = zip(*(
            tf.split(
                value=lstm,
                num_or_size_splits=2,
                axis=-1)
            for lstm
            in lstms))

        return tf.concat(
            values=fw+bw,
            axis=-1)

    def add_frobenius(self, *matrices):

        count = len(matrices)

        if count == 1:
            return tf.constant(
                value=0,
                dtype=tf.float32)

        return sum(
            tf.square(
                tf.norm(
                    tf.matmul(m, tf.transpose(m2)),
                    ord='fro',
                    axis=[0, 1]))
            for i, m in enumerate(matrices)
            for j, m2 in enumerate(matrices)
            if i < j) / (count * count-1 / 2)


    def add_task_layer(self, cont_repr, task, lang):
        layer_class = globals()[f'{task.upper()}Layer']
        return layer_class(self, cont_repr, task, lang)

    def lstm(self, inputs, sequence_lengths, cell_size, name_scope, avg_pool=False, dropout=True):
        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(cell_size)
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(cell_size)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=sequence_lengths,
                dtype=tf.float32)

            # shape = (batch_size, max(sequence_lengths), 2*cell_size)
            out = tf.concat(
                values=[out_fw, out_bw],
                axis=-1)

            if avg_pool:
                out = tf.reduce_sum(
                    input_tensor=out,
                    axis=1)
                div_mask = tf.cast(
                    x=sequence_lengths,
                    dtype=tf.float32)
                div_mask = tf.expand_dims(
                    input=div_mask,
                    axis=-1)  # Axis needed for broadcasting.
                out = out / div_mask

            if dropout:
                out = tf.nn.dropout(
                    x=out,
                    keep_prob=self.dropout)

            return out

    def create_sets(self, is_train=True, **kwargs):
        return [
            DatasetIterator(
                dataset=dt,
                config=self.config,
                is_train=is_train)
            for dt
            in self.dl.find(**kwargs)]

    def train:
        if have training set:
            train
        else:
            pass

    def evaluate:
        for st in eval_sets:
            results = st.layer.evaluate(st.iterator, st.dataset)
            results.update({
                'language': st.dataset.lang,
                'task': st.dataset.task,
                'role': st.dataset.role,
                'epoch': self.epoch
            })
            self.log(
                message={'results': results},
                level=LOG_RESULT)

        run all eval datasets

    def trainable(self):
        if exists train_set

    def get_representations(self):
        eval_sets = [
            st
            for st
            in self.create_sets()
            if st.dataset.role == 'train']

        return {
            st.dataset.lang: st.layer.get_representations(st.iterator, st.dataset)
            for st
            in eval_sets}

    def log(self, message, level):
        self.orch.log(message, level)