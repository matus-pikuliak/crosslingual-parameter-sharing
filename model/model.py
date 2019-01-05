import random
import datetime
import os

import tensorflow as tf
import numpy as np

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
        self.layers = {}

    def task_code(self, task, lang):
        if task == 'lmo':
            return f'{task}-{lang}'
        if self.config.task_layer_sharing:
            return task
        else:
            return f'{task}-{lang}'

    def _build_graph(self):
        self.add_inputs()
        self.add_utils()
        word_repr = self.add_word_processing()
        cont_repr = self.add_sentence_processing(word_repr)
        self.cont_repr_with_mask = tf.boolean_mask(cont_repr, self.sentence_lengths_mask)
        self.add_adversarial_loss(cont_repr)
        self.add_task_layers(cont_repr)


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

    def add_sentence_processing(self, word_repr):
        return self.lstm(
            inputs=word_repr,
            sequence_lengths=self.sentence_lengths,
            cell_size=self.config.word_lstm_size,
            name_scope='word_bilstm')

    def add_task_layers(self, cont_repr):

        for (task, lang) in self.config.tasks:
            task_code = self.task_code(task, lang)
            if task_code not in self.layers:
                layer_class = globals()[f'{task.upper()}Layer']
                new_layer = layer_class(self, task, lang, cont_repr)
                self.layers[task_code] = new_layer

    def add_utils(self):
        """
        Several useful nodes.
        """
        self.sentence_lengths_mask = tf.sequence_mask(self.sentence_lengths)
        self.total_batch_length = tf.reduce_sum(self.sentence_lengths)
        self.batch_size = tf.shape(self.word_ids)[0]
        self.max_length = tf.shape(self.word_ids)[1]

    def add_adversarial_loss(self, cont_repr):
        lambda_ = self.config.adversarial_lambda

        cont_repr = tf.boolean_mask(
            tensor=cont_repr,
            mask=self.sentence_lengths_mask)

        gradient_reversal = tf.stop_gradient((1+lambda_)*cont_repr) - lambda_*cont_repr

        hidden = tf.layers.dense(
            inputs=gradient_reversal,
            units=self.config.hidden_size,
            activation=tf.nn.relu)
        logits = tf.layers.dense(
            inputs=hidden,
            units=len(self.dl.langs))

        one_hot_lang = tf.one_hot(
            indices=self.lang_id,
            depth=len(self.langs))
        one_hot_lang = tf.expand_dims(
            input=one_hot_lang,
            axis=0)
        one_hot_lang = tf.tile(
            input=one_hot_lang,
            multiples=[tf.shape(cont_repr)[0], 1])

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_lang,
            logits=logits)
        self.adversarial_loss = tf.reduce_mean(loss)

    def add_train_op(self, loss):
        if self.config.adversarial_training and len(self.langs) > 1:
            use_adversarial = tf.less_equal(
                tf.random.uniform(shape=[]),
                1 / self.config.adversarial_freq)
            use_adversarial = tf.cast(
                x=use_adversarial,
                dtype=tf.float32)
            loss += self.adversarial_loss * use_adversarial
        return GeneralModel.add_train_op(self, loss)

    def run_epoch(self):

        if self.config.train_only is None:
            train_sets = self.create_sets(role='train')
            eval_sets = self.create_sets(is_train=False)
        else:
            task, lang = self.config.train_only.split('-')
            train_sets = self.create_sets(role='train', task=task, lang=lang)
            eval_sets = self.create_sets(is_train=False, task=task, lang=lang)

        for _ in range(self.config.epoch_steps * len(train_sets)):
            st = random.choice(train_sets)
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
            self.log(results, LOG_RESULT)

    def create_sets(self, is_train=True, **kwargs):
        return [
            DatasetIterator(
                dataset=dt,
                config=self.config,
                layer=self.layers[self.task_code(dt.task, dt.lang)],
                is_train=is_train)
            for dt
            in self.dl.find(**kwargs)]

    def lstm(self, inputs, sequence_lengths, cell_size, name_scope, avg_pool=False, dropout=True):
        with tf.variable_scope(name_scope):
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

    """
    temp_* methods are used for various experiments, but they are not essential for model itself.
    """

    def temp_export_representations(self, task, lang, role, sample_size):

        dt = self.dl.find_one(task=task, lang=lang, role=role)
        ite = DatasetIterator(
            dataset=dt,
            config=self.config,
            layer=self.layers[self.task_code(dt.task, dt.lang)],
            is_train=False)


        cont_repr = None
        cont_repr_grad = None
        cont_repr_weights_grad = None
        global_norm = 0
        batch_count = 0

        for batch in ite.iterator:
            batch_count += 1

            *_, desired_labels, desired_arcs = batch
            fd = ite.layer.test_feed_dict(batch, dt)
            fd.update({
                ite.layer.desired_arcs.placeholder: desired_arcs,
                ite.layer.desired_labels.placeholder: desired_labels
            })
            fetches = self.sess.run(
                fetches=[
                    self.cont_repr_with_mask,
                    ite.layer.cont_repr_grad,
                    ite.layer.global_norm,
                    ite.layer.grads[ite.layer.cont_repr_weights]
                ],
                feed_dict=fd)

            global_norm += fetches[2]

            if cont_repr is None:
                cont_repr_weights_grad = fetches[3]
                cont_repr = fetches[0]
                cont_repr_grad = fetches[1]
            else:
                cont_repr_weights_grad += fetches[3]
                if cont_repr.shape[0] < 50_000:
                    cont_repr = np.vstack((cont_repr, fetches[0]))
                    cont_repr_grad = np.vstack((cont_repr_grad, fetches[1]))

        return {
            'cont_repr': cont_repr,
            'cont_repr_grad': cont_repr_grad,
            'gradient_norm': global_norm / batch_count,
            'cont_repr_weights_grad': cont_repr_weights_grad / batch_count,
            'cont_repr_weights': self.sess.run(ite.layer.cont_repr_weights)
        }
