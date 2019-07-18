import types

import numpy as np
import tensorflow as tf

from constants import LOG_RESULT
from data.embedding import Embeddings
from model.dataset_iterator import DatasetIterator



class Model:

    def __init__(self, task, lang, orchestrator, data_loader, config, logger):
        self.task = task
        self.lang = lang
        self.orch = orchestrator
        self.dl = data_loader
        self.config = config
        self.logger = logger

        self.train_set = self.create_sets(task=self.task, lang=self.lang, role='train')[0]
        self.eval_sets = self.create_sets(is_train=False, task=self.task, lang=self.lang)

        self.n = types.SimpleNamespace()

    @staticmethod
    def factory(task, *args):
        from model.model_dep import ModelDEP
        from model.model_lmo import ModelLMO
        from model.model_ner import ModelNER
        from model.model_pos import ModelPOS

        classes = {
            'pos': ModelPOS,
            'ner': ModelNER,
            'lmo': ModelLMO,
            'dep': ModelDEP
        }
        return classes[task](task, *args)

    def __repr__(self):
        return f'{self.task}-{self.lang}_Model'

    def create_sets(self, is_train=True, **kwargs):
        return [
            DatasetIterator(
                dataset=dt,
                config=self.config,
                is_train=is_train)
            for dt
            in self.dl.find(**kwargs)]

    def build_graph(self):
        self.add_inputs()
        self.add_utils()
        self.add_word_processing()
        self.add_sentence_processing()
        self.add_task_layer()
        self.add_unit_strength()
        self.add_train_op()
        self.metrics = self.add_metrics()

        self.check_nodes()

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

        with tf.variable_scope('word_embeddings', reuse=tf.AUTO_REUSE):
            word_emb_matrix = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.cast(self.load_embeddings(self.lang), tf.float32),
                trainable=self.config.train_emb,
                name=f'word_embedding_matrix_{self.lang}')

        word_embeddings = tf.nn.embedding_lookup(
            params=word_emb_matrix,
            ids=self.n.word_ids)

        if self.config.char_level:
            char_embeddings = self.add_char_embeddings()
            word_embeddings = tf.concat(
                values=[word_embeddings, char_embeddings],
                axis=-1)

        word_embeddings = tf.nn.dropout(
            x=word_embeddings,
            keep_prob=self.orch.n.dropout)

        self.n.word_embeddings = word_embeddings

    def load_embeddings(self, lang):
        # FIXME: ako casto sa toto spusta pri multitask?
        emb = Embeddings(lang, self.config)
        return emb.matrix(self.dl.lang_vocabs[lang])

    def add_char_embeddings(self):
        #FIXME: add char level sharing strategies
        with tf.variable_scope('character_embeddings', reuse=tf.AUTO_REUSE):
            emb_matrix = tf.get_variable(
                dtype=tf.float32,
                shape=(len(self.dl.char_vocab), self.config.char_emb_size),
                name="char_embeddings")

        char_embeddings = tf.nn.embedding_lookup(
            params=emb_matrix,
            ids=self.n.char_ids,
            name="char_embeddings_lookup")

        shape = tf.shape(char_embeddings)
        max_sentence, max_word = shape[1], shape[2]

        char_embeddings = tf.reshape(
            tensor=char_embeddings,
            shape=[-1, max_word, self.config.char_emb_size])
        word_lengths = tf.reshape(
            tensor=self.n.word_lengths,
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

    def add_sentence_processing(self):

        lstms = []

        def get_lstm(name_scope):
            return self.lstm(
                inputs=self.n.word_embeddings,
                sequence_lengths=self.n.sentence_lengths,
                cell_size=self.config.word_lstm_size,
                name_scope=name_scope)

        if self.config.word_lstm_private:
            lstms.append(get_lstm(f'word_bilstm_{self.task}_{self.lang}'))

        if self.config.word_lstm_task:
            lstms.append(get_lstm(f'word_bilstm_{self.task}'))

        if self.config.word_lstm_lang:
            lstms.append(get_lstm(f'word_bilstm_{self.lang}'))

        if self.config.word_lstm_global:
            lstms.append(get_lstm(f'word_bilstm'))

        assert(len(lstms) > 0)

        self.n.frobenius = self.add_frobenius(*(
            tf.boolean_mask(
                tensor=lstm,
                mask=self.n.sentence_lengths_mask)
            for lstm
            in lstms))

        if len(lstms) == 1:
            self.n.contextualized = lstms[0]
        else:
            fw, bw = zip(*(
                tf.split(
                    value=lstm,
                    num_or_size_splits=2,
                    axis=-1)
                for lstm
                in lstms))

            self.n.contextualized = tf.concat(
                values=fw+bw,
                axis=-1)

        self.n.contextualized_masked = tf.boolean_mask(
                tensor=self.n.contextualized,
                mask=self.n.sentence_lengths_mask)

        self.add_adversarial_loss()

    def add_frobenius(self, *matrices):

        count = len(matrices)

        if count == 1:
            return tf.constant(
                value=0,
                dtype=tf.float32)

        return sum(
            tf.square(
                tf.norm(
                    tf.matmul(m, tf.transpose(n)), # FIXME: je spravny clen transponovany?
                    ord='fro',
                    axis=[0, 1]))
            for i, m in enumerate(matrices)
            for j, n in enumerate(matrices)
            if i < j) / (count * count-1 / 2)

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
                    keep_prob=self.orch.n.dropout)

            return out

    def add_adversarial_loss(self):
        with tf.variable_scope('adversarial_training', reuse=tf.AUTO_REUSE):
            cont_repr = self.n.contextualized_masked
            lambda_ = self.config.adversarial_lambda
            gradient_reversal = tf.stop_gradient((1 + lambda_) * cont_repr) - lambda_ * cont_repr

            hidden = tf.layers.dense(
                inputs=gradient_reversal,
                units=self.config.hidden_size,
                activation=tf.nn.relu)
            logits = tf.layers.dense(
                inputs=hidden,
                units=len(self.orch.langs))

            one_hot_lang = tf.one_hot(
                indices=self.n.lang_id,
                depth=len(self.orch.langs))
            one_hot_lang = tf.expand_dims(
                input=one_hot_lang,
                axis=0)
            one_hot_lang = tf.tile(
                input=one_hot_lang,
                multiples=[tf.shape(cont_repr)[0], 1])

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot_lang,
                logits=logits)

            self.n.adversarial_loss = tf.reduce_mean(loss)

            use_adversarial = tf.less_equal(
                tf.random.uniform(shape=[]),
                1 / self.config.adversarial_freq)

            use_adversarial = tf.cast(
                x=use_adversarial,
                dtype=tf.float32)

            self.n.adversarial_term = self.n.adversarial_loss * use_adversarial

    def add_task_layer(self):
        raise NotImplementedError

    def add_unit_strength(self):

        repr = self.n.contextualized_masked
        norms = tf.norm(
            tensor=self.n.contextualized_weights,
            axis=1)
        norms = tf.expand_dims(
            input=norms,
            axis=0)
        norms = tf.tile(
            input=norms,
            multiples=[self.n.total_batch_length, 1])

        norms = tf.abs(tf.multiply(norms, repr))
        norms = tf.divide(
            norms,
            tf.reduce_sum(
                input_tensor=norms,
                axis=1,
                keepdims=True
            ))

        self.n.unit_strength = tf.reduce_mean(
            input_tensor=norms,
            axis=0)

    def add_metrics(self):
        raise NotImplementedError

    def add_train_op(self):
        total_loss = self.n.loss
        if self.config.adversarial_training and len(self.orch.langs) > 1:
            total_loss += self.n.adversarial_term
        if self.config.frobenius > 0:
            total_loss += self.config.frobenius * self.n.frobenius
        grads, vs = zip(*self.orch.n.optimizer.compute_gradients(total_loss))
        gradient_norm = tf.global_norm(grads)
        if self.config.clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)

        self.n.train_op, self.n.gradient_norm = self.orch.n.optimizer.apply_gradients(zip(grads, vs)), gradient_norm

    def check_nodes(self):
        ...
        # FIXME: add all nodes that should be defined in the model

    def basic_feed_dict(self, batch):
        word_ids, sentence_lengths, char_ids, word_lengths, *_ = batch

        return {
            self.n.word_ids: word_ids,
            self.n.sentence_lengths: sentence_lengths,
            self.n.char_ids: char_ids,
            self.n.word_lengths: word_lengths,
            self.n.lang_id: self.orch.langs.index(self.lang),
            self.n.task_id: self.orch.tasks.index(self.task)
        }

    def train_feed_dict(self, batch):
        fd = self.basic_feed_dict(batch)
        fd.update({
            self.orch.n.learning_rate: self.orch.current_learning_rate(),
            self.orch.n.dropout: self.config.dropout
        })
        return fd

    def test_feed_dict(self, batch):
        fd = self.basic_feed_dict(batch)
        fd.update({
            self.orch.n.dropout: 1
        })
        return fd

    def train_step(self):
        batch = next(self.train_set.iterator)
        fd = self.train_feed_dict(batch)
        self.orch.sess.run(
            fetches=self.n.train_op,
            feed_dict=fd)

    def evaluate(self):
        for eval_set in self.eval_sets:
            fetches = self.basic_fetches()
            fetches.update(self.metrics)

            results = self.evaluate_batches_iterator(
                iterator=eval_set.iterator,
                fetch_nodes=list(fetches.values()))

            results = dict(zip(
                fetches.keys(),
                zip(*results)
            ))

            output = self.basic_results(results)
            output.update(self.evaluate_task(results))
            output.update({
                'language': self.lang,
                'task': self.task,
                'role': eval_set.dataset.role,
                'epoch': self.orch.epoch
            })
            self.log(
                message={'results': output},
                level=LOG_RESULT)

    def evaluate_batches_iterator(self, iterator, fetch_nodes):
        for batch in iterator:
            yield self.orch.sess.run(
                fetches=fetch_nodes,
                feed_dict=self.test_feed_dict(batch))

    def get_representations(self, iterator):
        return np.vstack([
            self.orch.sess.run(
                fetches=self.n.contextualized_masked,
                feed_dict=self.test_feed_dict(next(iterator)))
            for _ in range(5)
        ])

    def basic_fetches(self):
        return {
            'loss': self.n.loss,
            'adv_loss': self.n.adversarial_loss,
            'gradient_norm': self.n.gradient_norm,
            'length': self.n.total_batch_length,
            'unit_strength': self.n.unit_strength,
            'frobenius': self.n.frobenius
        }

    def basic_results(self, results):
        return {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
            'gradient_norm': np.mean(results['gradient_norm']),
            'unit_strength_2': np.std(np.mean(results['unit_strength'], axis=0)),
            'frobenius': np.mean(results['frobenius'])
        }

    def task_code(self):
        if self.task == 'lmo':
            return f'{self.task}-{self.lang}'
        if self.config.task_layer_sharing:
            return self.task
        else:
            return f'{self.task}-{self.lang}'

    def trainable(self):
        return self.train_set is not None

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