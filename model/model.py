import itertools
import random

import numpy as np
import tensorflow as tf

from data.embedding import Embeddings
from model.dataset_iterator import DatasetIterator
from model.general_model import GeneralModel

from constants import LOG_CRITICAL, LOG_MESSAGE, LOG_RESULT

from model.layer_ner import NERLayer
from model.layer_pos import POSLayer
from model.layer_dep import DEPLayer
from model.layer_lmo import LMOLayer

class Model(GeneralModel):

    def __init__(self, *args, **kwargs):
        GeneralModel.__init__(self, *args, **kwargs)
        self.langs = self.dl.langs
        self.tasks = self.dl.tasks
        self.task_langs = itertools.product(self.tasks, self.langs)

    def _build_graph(self):
        self.add_inputs()
        self.add_utils()
        word_repr = self.add_word_processing()
        self.layers = {}
        for (task, lang) in self.task_langs:
            cont_repr = self.add_sentence_processing(word_repr, task, lang)
            self.layers[task, lang] = self.add_task_layer(cont_repr, task, lang)

    def add_inputs(self):
        # shape = (batch size, max_sentence_length)
        self.word_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="word_ids")

        # shape = (batch size)
        self.sentence_lengths = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='sentence_lengths')

        # shape = (batch_size, max_sentence_length, max_word_length)
        self.char_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None, None],
            name='char_ids')

        # shape = (batch_size, max_sentence_length)
        self.word_lengths = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='word_lengths')

        self.lang_id = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name='lang_id')

        self.lang_flags = {
            lang: tf.equal(self.lang_id, i)
            for i, lang
            in enumerate(self.langs)
        }

        self.task_id = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name='task_id')

        self.task_flags = {
            task: tf.equal(self.task_id, i)
            for i, task
            in enumerate(self.tasks)
        }

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

        if not self.config.private_params:
            return self.lstm(
                inputs=word_repr,
                sequence_lengths=self.sentence_lengths,
                cell_size=self.config.word_lstm_size,
                name_scope='word_bilstm')

        else:
            shared_lstm = self.lstm(
                inputs=word_repr,
                sequence_lengths=self.sentence_lengths,
                cell_size=self.config.word_lstm_size - self.config.private_size,
                name_scope='word_bilstm')

            private_lstm = self.lstm(
                    inputs=word_repr,
                    sequence_lengths=self.sentence_lengths,
                    cell_size=self.config.private_size,
                    name_scope=f'word_bilstm_{task}-{lang}')

            shared_fw, shared_bw = tf.split(
                value=shared_lstm,
                num_or_size_splits=2,
                axis=-1)

            private_fw, private_bw = tf.split(
                value=private_lstm,
                num_or_size_splits=2,
                axis=-1)

            return tf.concat(
                values=(shared_fw, private_fw, shared_bw, private_bw),
                axis=-1)

    def add_task_layer(self, cont_repr, task, lang):
        layer_class = globals()[f'{task.upper()}Layer']
        return layer_class(self, cont_repr, task, lang)

    def add_utils(self):
        """
        Several useful nodes.
        """
        self.sentence_lengths_mask = tf.sequence_mask(self.sentence_lengths)
        self.total_batch_length = tf.reduce_sum(self.sentence_lengths)
        self.batch_size = tf.shape(self.word_ids)[0]
        self.max_length = tf.shape(self.word_ids)[1]

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

    def run_epoch(self):

        if self.config.train_only is None:
            train_sets = self.create_sets(role='train')
            if self.config.focus_on is None:
                eval_sets = self.create_sets(is_train=False)
            else:
                task, lang = self.config.focus_on.split('-')
                eval_sets = self.create_sets(is_train=False, task=task, lang=lang)
        else:
            task, lang = self.config.train_only.split('-')
            train_sets = self.create_sets(role='train', task=task, lang=lang)
            eval_sets = self.create_sets(is_train=False, task=task, lang=lang)

        for _ in range(self.config.epoch_steps * len(train_sets)):

            if self.config.focus_on is None:
                st = np.random.choice(train_sets)
            else:
                task, lang = self.config.focus_on.split('-')

                def on_off_prob(focus_rate, count):
                    on = focus_rate if count > 1 else 1
                    off = (1 - focus_rate) / (count - 1) if count > 1 else 0
                    return on, off

                on_task_prob, off_task_prob = on_off_prob(self.config.focus_rate, len(self.tasks))
                on_lang_prob, off_lang_prob = on_off_prob(self.config.focus_rate, len(self.langs))
                task_probs = [
                    (
                        (on_task_prob if st.dataset.task == task else off_task_prob) *
                        (on_lang_prob if st.dataset.lang == lang else off_lang_prob)
                    ) for st in train_sets]
                st = np.random.choice(train_sets, p=task_probs)

            st.layer.train(next(st.iterator), st.dataset)

        self.log(f'Epoch {self.epoch} training done.', LOG_MESSAGE)

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

    def evaluate_epoch(self):
        eval_sets = self.create_sets(is_train=False)

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

    def create_sets(self, is_train=True, **kwargs):
        return [
            DatasetIterator(
                dataset=dt,
                config=self.config,
                layer=self.layers[dt.task, dt.lang],
                is_train=is_train)
            for dt
            in self.dl.find(**kwargs)]
