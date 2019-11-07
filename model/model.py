import types
import itertools

import numpy as np
import tensorflow as tf

from constants import LOG_RESULT, TASKS
from model.dataset_iterator import DatasetIterator


class Model:

    def __init__(self, task, lang, orchestrator, data_loader, config, logger):
        self.task = task
        self.lang = lang
        self.orch = orchestrator
        self.dl = data_loader
        self.config = config
        self.logger = logger
        self.previous_results = dict()

        self.n = types.SimpleNamespace()

    @staticmethod
    def factory(task, *args):

        def model_class(task):
            return f'Model{task.upper()}'

        classes = {
            task: __import__(f'model.model_{task}', fromlist=[model_class(task)])
            for task
            in TASKS
        }

        return getattr(classes[task], model_class(task))(task, *args)

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

    def train_set(self):
        return self.create_sets(task=self.task, lang=self.lang, role='train')[0]

    def eval_sets(self):
        return self.create_sets(is_train=False, task=self.task, lang=self.lang)

    def build_graph(self):
        self.add_inputs()
        self.add_utils()
        self.add_word_processing()
        self.add_sentence_processing()
        self.add_task_layer()
        self.add_unit_strength()
        self.add_train_op()
        self.metrics = self.add_metrics()

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
        embs = list()

        if self.config.word_level:
            embs.append(self.add_pretrained_embeddings())
        if self.config.char_level:
            embs.append(self.add_char_embeddings())
        assert len(embs) > 0
        model_embeddings = self.add_model_embeddings()
        if model_embeddings is not None:
            embs.append(model_embeddings)

        if len(embs) > 1:
            word_embeddings = tf.concat(
                values=embs,
                axis=-1)
        else:
            word_embeddings = embs[0]

        word_embeddings = tf.nn.dropout(
            x=word_embeddings,
            rate=self.orch.n.dropout)

        self.n.word_embeddings = word_embeddings

    def add_pretrained_embeddings(self):
        word_emb_matrix = self.orch.embeddings(self.lang)

        return tf.nn.embedding_lookup(
            params=word_emb_matrix,
            ids=self.n.word_ids)

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

        char_embeddings = tf.reshape(
            tensor=char_embeddings,
            shape=[-1, tf.shape(char_embeddings)[2], self.config.char_emb_size])
        word_lengths = tf.reshape(
            tensor=self.n.word_lengths,
            shape=[-1])

        lstms = []

        def get_lstm(name_scope):
            return self.lstm(
                inputs=char_embeddings,
                sequence_lengths=word_lengths,
                cell_size=self.config.char_lstm_size,
                name_scope=name_scope,
                avg_pool=True)

        if self.config.char_lstm_private:
            lstms.append(get_lstm(f'char_bilstm_{self.task}_{self.lang}'))

        if self.config.char_lstm_lang:
            lstms.append(get_lstm(f'char_bilstm_{self.lang}'))

        if self.config.char_lstm_global:
            lstms.append(get_lstm(f'char_bilstm'))

        assert(len(lstms) > 0)

        if len(lstms) == 1:
            char_lstm_out = lstms[0]
        else:
            char_lstm_out = tf.concat(
                values=lstms,
                axis=-1)

        char_lstm_out = tf.reshape(
            tensor=char_lstm_out,
            shape=[self.n.batch_size, self.n.max_length, 2 * len(lstms) * self.config.char_lstm_size])
        char_lstm_out = tf.where(
            condition=tf.is_nan(char_lstm_out),
            x=tf.zeros_like(char_lstm_out),
            y=char_lstm_out)

        return char_lstm_out

    def add_model_embeddings(self):

        embs = list()

        with tf.variable_scope('model_embeddings', reuse=tf.AUTO_REUSE):

            task_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=(len(self.orch.tasks), 25),
                name="task_embeddings")

            lang_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=(len(self.orch.langs), 25),
                name="lang_embeddings")

            task_lang_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=(len(self.orch.tasks) * len(self.orch.langs), 25),
                name="task_lang_embeddings")

        if self.config.emb_task:
            embs.append(tf.nn.embedding_lookup(
                params=task_embeddings,
                ids=self.n.task_id))

        if self.config.emb_lang:
            embs.append(tf.nn.embedding_lookup(
                params=lang_embeddings,
                ids=self.n.lang_id))

        if self.config.emb_task_lang:
            embs.append(tf.nn.embedding_lookup(
                params=task_lang_embeddings,
                ids=self.n.task_id * len(self.orch.tasks) + self.n.lang_id))

        if not embs:
            return None

        embs = tf.concat(
            values=embs,
            axis=-1)
        embs = tf.expand_dims(
            input=embs,
            axis=0)
        embs = tf.expand_dims(
            input=embs,
            axis=0)
        return tf.tile(
            input=embs,
            multiples=(self.n.batch_size, self.n.max_length, 1))

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

        self.n.ortho = self.add_ortho(*(
            tf.boolean_mask(
                tensor=lstm,
                mask=self.n.sentence_lengths_mask)
            for lstm
            in lstms))

        if len(lstms) == 1:
            contextualized = lstms[0]
        else:
            fw, bw = zip(*(
                tf.split(
                    value=lstm,
                    num_or_size_splits=2,
                    axis=-1)
                for lstm
                in lstms))

            contextualized = tf.concat(
                values=fw+bw,
                axis=-1)

        self.n.contextualized_masked_no_dropout = tf.boolean_mask(
                tensor=contextualized,
                mask=self.n.sentence_lengths_mask)

        self.add_adversarial_loss()

        self.n.contextualized = tf.nn.dropout(
            x=contextualized,
            rate=self.orch.n.dropout)

    def add_ortho(self, *matrices):

        if len(matrices) == 1:
            return tf.constant(
                value=0,
                dtype=tf.float32)

        matrices = [tf.math.l2_normalize(
            x=matrix,
            axis=1)
            for matrix
            in matrices]

        losses = []
        for m1, m2 in itertools.combinations(matrices, 2):
            loss = tf.matmul(
                a=m1,
                b=m2,
                transpose_b=True)
            loss = tf.square(
                tf.norm(
                    tensor=loss,
                    ord='fro',
                    axis=[0, 1]))
            losses.append(loss)

        return sum(losses) / tf.cast(
            x=len(losses) * self.n.total_batch_length**2,
            dtype=tf.float32)

    def lstm(self, inputs, sequence_lengths, cell_size, name_scope, avg_pool=False):
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

            return out

    def add_adversarial_loss(self):
        with tf.variable_scope('adversarial_training', reuse=tf.AUTO_REUSE):
            cont_repr = self.n.contextualized_masked_no_dropout
            lambda_ = self.config.adversarial_lambda
            gradient_reversal = tf.stop_gradient((1 + lambda_) * cont_repr) - lambda_ * cont_repr

            hidden = tf.layers.dense(
                inputs=gradient_reversal,
                units=self.config.hidden_size,
                activation=tf.nn.relu)

            def loss(id, len_ids):
                logits = tf.layers.dense(
                    inputs=hidden,
                    units=len_ids)

                one_hot = tf.one_hot(
                    indices=id,
                    depth=len_ids)
                one_hot = tf.expand_dims(
                    input=one_hot,
                    axis=0)
                one_hot = tf.tile(
                    input=one_hot,
                    multiples=[self.n.total_batch_length, 1])

                out = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=one_hot,
                    logits=logits)
                return tf.reduce_mean(out)

            lang_loss = loss(self.n.lang_id, len(self.orch.langs))
            task_loss = loss(self.n.task_id, len(self.orch.tasks))

            self.n.adversarial_loss = lang_loss + task_loss

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

        repr = self.n.contextualized_masked_no_dropout
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
        if self.config.ortho > 0:
            total_loss += self.config.ortho * self.n.ortho
        grads, vs = zip(*self.orch.n.optimizer.compute_gradients(total_loss))
        gradient_norm = tf.global_norm(grads)
        if self.config.clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)

        self.n.train_op, self.n.gradient_norm = self.orch.n.optimizer.apply_gradients(zip(grads, vs)), gradient_norm

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
            self.orch.n.dropout: 0
        })
        return fd

    def train_step(self):
        batch = next(self.train_set().iterator)
        fd = self.train_feed_dict(batch)
        self.orch.sess.run(
            fetches=self.n.train_op,
            feed_dict=fd)

    def evaluate(self):
        for eval_set in self.eval_sets():
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

            if eval_set.dataset.role == 'dev':
                self.previous_results[self.orch.epoch] = self.get_latest_result(output)
            self.log(
                message={'results': output},
                level=LOG_RESULT)

    def evaluate_batches_iterator(self, iterator, fetch_nodes):
        for batch in iterator:
            yield self.orch.sess.run(
                fetches=fetch_nodes,
                feed_dict=self.test_feed_dict(batch))

    def should_early_stop(self):
        if self.config.early_stopping >= len(self.previous_results):
            return False
        if self.get_best_epoch() > len(self.previous_results) - self.config.early_stopping:
            return False
        return True

    def get_representations(self, batch_count):
        test_sets = self.create_sets(is_train=False, role='test', task=self.task, lang=self.lang)
        iterator = test_sets[0].iterator

        return np.vstack([
            self.orch.sess.run(
                fetches=self.n.contextualized_masked_no_dropout,
                feed_dict=self.test_feed_dict(next(iterator)))
            for _ in range(batch_count)
        ])

    def basic_fetches(self):
        return {
            'loss': self.n.loss,
            'adv_loss': self.n.adversarial_loss,
            'gradient_norm': self.n.gradient_norm,
            'length': self.n.total_batch_length,
            'unit_strength': self.n.unit_strength,
            'ortho': self.n.ortho,
        }

    def basic_results(self, results):
        return {
            'loss': np.mean(results['loss']),
            'adv_loss': np.mean(results['adv_loss']),
            'gradient_norm': np.mean(results['gradient_norm']),
            'unit_strength': np.std(np.mean(results['unit_strength'], axis=0)),
        }

    def task_code(self):
        if self.task == 'lmo':
            return f'{self.task}-{self.lang}'
        if self.config.task_layer_sharing:
            return self.task
        else:
            return f'{self.task}-{self.lang}'

    def is_focused(self):
        if self.config.focus_on is not None:
            if [self.task, self.lang] == self.config.focus_on.split('-'):
                return True
        return False

    def task_layer_scope(self):
        if self.config.task_layer_private:
            return f'{self.task}-{self.lang}'
        return self.task

    def log(self, message, level):
        self.orch.log(message, level)