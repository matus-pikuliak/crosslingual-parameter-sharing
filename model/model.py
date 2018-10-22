import tensorflow as tf
import numpy as np
import datetime
import os
from logs.logger_init import LoggerInit
from model.general_model import GeneralModel


class Model(GeneralModel):

    def task_code(self, task, lang):
        if self.config.crf_sharing:
            return task
        else:
            return task + lang

    def load_embeddins(self, lang):
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

    def add_crf(self, task, task_code):
        with tf.variable_scope(task_code):
            tag_count = len(self.dl.task_vocabs[task])
            max_length = tf.shape(self.word_ids)[1]

            output = tf.reshape(self.word_lstm_output, [-1, 2 * self.config.word_lstm_size])
            W = tf.get_variable(dtype=self.type, shape=[2 * self.config.word_lstm_size, tag_count],
                                name="weights")
            b = tf.get_variable(dtype=self.type, shape=[tag_count], initializer=tf.zeros_initializer(),
                                name="biases")
            output = tf.matmul(output, W) + b
            self.predicted_labels[task_code] = tf.reshape(output, [-1, max_length, tag_count])

            # expected output
            # shape = (batch_size, max_length)
            self.true_labels[task_code] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="labels")

            # loss
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.predicted_labels[task_code],
                self.true_labels[task_code],
                self.sequence_lengths)
            self.trans_params[task_code] = trans_params  # need to evaluate it for decoding
            self.loss[task_code] = tf.reduce_mean(-log_likelihood)

        # training
        # Rremoving this part from task scope lets the graph reuse optimizer parameters
        # FIXME: Is that so?
        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def add_dep(self, task_code):
        with tf.variable_scope(task_code):
            root = tf.get_variable("root_vector", dtype=self.type, shape=[2*self.config.word_lstm_size])  # dim
            root = tf.expand_dims(root, 0)
            root = tf.expand_dims(root, 0)
            root = tf.tile(
                root,
                [tf.shape(self.word_lstm_output)[0], 1, 1]
            )

            words = self.word_lstm_output
            words_root = tf.concat([root, words], 1)

            tile_a = tf.tile(
                tf.expand_dims(words, 2),
                [1, 1, tf.shape(words_root)[1], 1]
            )
            tile_b = tf.tile(
                tf.expand_dims(words_root, 1),
                [1, tf.shape(words)[1], 1, 1]
            )

            combinations = tf.concat([tile_a, tile_b], axis=3)
            combinations = tf.reshape(combinations, [-1, 4*self.config.word_lstm_size])

            hidden = 500

            W = tf.get_variable("W", dtype=self.type, shape=[4*self.config.word_lstm_size, hidden])
            b = tf.get_variable("b", dtype=self.type, shape=[hidden])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[hidden, 1])

            seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths), shape=[-1])

            combinations = tf.nn.tanh(tf.matmul(combinations, W) + b)
            all_combinations = combinations
            combinations = tf.matmul(combinations, W2)
            combinations = tf.reshape(combinations, [-1, tf.shape(words_root)[1]])  # (batch_size x length) x length+1 (root)

            self.arc_ids = tf.placeholder(tf.int64, shape=[None, None])  # batch size x length
            _arc_ids = tf.reshape(self.arc_ids, [-1])
            _arc_ids = tf.boolean_mask(_arc_ids, seq_mask)

            predicted_arc_ids = tf.argmax(combinations, axis=1)
            _predicted_arc_ids = tf.reshape(predicted_arc_ids, tf.shape(self.arc_ids))

            self.golden_arc_ids = tf.placeholder(tf.bool, shape=[])
            relevant_arc_ids = tf.cond(self.golden_arc_ids, lambda: self.arc_ids, lambda: _predicted_arc_ids)

            relevant_arc_ids = tf.one_hot(relevant_arc_ids, tf.shape(words_root)[1], on_value=True, off_value=False, dtype=tf.bool)
            relevant_arc_ids = tf.reshape(relevant_arc_ids, [-1])
            relevant_arcs = tf.boolean_mask(all_combinations, relevant_arc_ids)
            relevant_arcs = tf.boolean_mask(relevant_arcs, seq_mask)

            self.true_labels[task_code] = tf.placeholder(tf.int64, shape=[None, None], name="labels")
            labels = tf.reshape(self.true_labels[task_code], [-1])
            labels = tf.boolean_mask(labels, seq_mask)
            one_hot_labels = tf.one_hot(labels, len(self.dl.task_vocabs['dep']))

            W3 = tf.get_variable("W3", dtype=self.type, shape=[hidden, len(self.dl.task_vocabs['dep'])])
            b3 = tf.get_variable("b3", dtype=self.type, shape=[len(self.dl.task_vocabs['dep'])])
            predicted_arc_labels = tf.matmul(relevant_arcs, W3) + b3
            predicted_arc_labels_ids = tf.argmax(predicted_arc_labels, axis=1)
            loss_las = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot_labels,
                logits=predicted_arc_labels,
                dim=-1,
            ))


            predicted_arc_ids = tf.boolean_mask(predicted_arc_ids, seq_mask)
            uas = tf.equal(predicted_arc_ids, _arc_ids)
            las = tf.logical_and(tf.equal(predicted_arc_labels_ids, labels), uas)
            self.uas = tf.reduce_sum(tf.count_nonzero(uas))
            self.las = tf.reduce_sum(tf.count_nonzero(las))

            combinations = tf.boolean_mask(combinations, seq_mask)
            arc_one_hots = tf.one_hot(_arc_ids, tf.shape(words_root)[1])  # length+1
            loss_uas = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=arc_one_hots,
                logits=combinations,
                dim=-1,
            ))
            self.loss[task_code] = loss_las + loss_uas

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def add_nli(self, task_code):
        with tf.variable_scope(task_code):
            words = self.word_lstm_output
            reduction = tf.concat([
                tf.reduce_max(words, axis=1),
                tf.reduce_mean(words, axis=1)
            ], axis=1)
            reduction = tf.reshape(reduction, [-1, 8 * self.config.word_lstm_size])
            premise, hypothesis = tf.split(reduction, [4 * self.config.word_lstm_size, 4 * self.config.word_lstm_size], axis=1)
            representation = tf.concat([
                tf.multiply(premise, hypothesis),
                premise - hypothesis
            ], axis=1)
            W = tf.get_variable("W", dtype=self.type, shape=[8 * self.config.word_lstm_size, 500])
            b = tf.get_variable("b", dtype=self.type, shape=[500])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[500, len(self.dl.task_vocabs['nli'])])

            representation = tf.matmul(representation, W) + b
            representation = tf.matmul(tf.nn.tanh(representation), W2)

            self.true_labels[task_code] = tf.placeholder(tf.int64, shape=[None],
                                                  name="labels")
            true_labels_one_hot = tf.one_hot(self.true_labels[task_code], depth=len(self.dl.task_vocabs['nli']))
            self.loss[task_code] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=true_labels_one_hot,
                logits=representation,
                dim=-1,
            ))

            predicted_labels = tf.argmax(representation, axis=1)
            correct_labels = tf.equal(predicted_labels, self.true_labels[task_code])
            self.correct_labels_count = tf.reduce_sum(tf.cast(correct_labels, dtype=tf.int32))

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

    def add_lmo(self, task_code, lang):
        with tf.variable_scope(task_code):
            max_len = tf.reduce_max(self.sequence_lengths)
            batch_size = tf.size(self.sequence_lengths)
            vocab_size = min(self.config.lmo_vocab_size + 1, len(self.dl.lang_vocabs[lang])) # +1 <unk>

            start_vec = tf.get_variable('start_vec', shape=[self.config.word_lstm_size], dtype=self.type)
            start_vec = tf.expand_dims(start_vec, 0)
            start_vec = tf.tile(start_vec, [batch_size, 1])
            start_vec = tf.expand_dims(start_vec, 1)
            _fd, _ = tf.split(self.lstm_fw, [max_len - 1, 1], axis=1)
            _fd = tf.concat([start_vec, _fd], 1)

            end_vec = tf.get_variable('end_vec', shape=[self.config.word_lstm_size], dtype=self.type)
            end_vec = tf.expand_dims(end_vec, 0)
            end_vec = tf.tile(end_vec, [batch_size, 1])
            end_vec = tf.expand_dims(end_vec, 1)
            one_hot = tf.one_hot(self.sequence_lengths - 1, max_len)
            one_hot = tf.expand_dims(one_hot, 2)
            end_vec = tf.matmul(one_hot, end_vec)

            _, _bd = tf.split(self.lstm_fw, [1, max_len - 1], axis=1)
            zeros = tf.zeros([batch_size, 1, self.config.word_lstm_size], dtype=self.type)
            _bd = tf.concat([_bd, zeros], 1)
            _bd = _bd + end_vec

            rp = tf.concat([_fd, _bd], axis=2)
            rp = tf.reshape(rp, [-1, 2 * self.config.word_lstm_size])
            seq_mask = tf.reshape(tf.sequence_mask(self.sequence_lengths, max_len), [-1])
            rp = tf.boolean_mask(rp, seq_mask)
            _ids = tf.boolean_mask(tf.reshape(self.word_ids, [-1]), seq_mask)
            _ids = tf.where(
                tf.less(_ids, vocab_size),
                _ids,
                tf.zeros(tf.shape(_ids), dtype=tf.int32)
            )
            # if id is more than vocab size, it is set to <unk> id = 0

            W = tf.get_variable("W", dtype=self.type, shape=[2 * self.config.word_lstm_size, 500])
            b = tf.get_variable("b", dtype=self.type, shape=[500])
            W2 = tf.get_variable("W2", dtype=self.type, shape=[500, vocab_size])

            rp = tf.matmul(rp, W) + b
            rp = tf.matmul(tf.nn.tanh(rp), W2)

            self.loss[task_code] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=_ids,
                logits=rp,
            ))

        self.train_op[task_code] = self.add_train_op(self.loss[task_code], task_code)

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
                initializer=tf.cast(self.load_embeddins(lang), self.type),
                trainable=self.config.train_emb,
                name="word_embeddings_%s" % lang)


        lambdas = lambda lang: lambda: _word_embeddings[lang]
        cases = dict([(self.language_flags[lang], lambdas(lang)) for lang in self.dl.langs()])
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

                    word_ids, sentence_lengths, char_ids, word_lengths, label_ids = next(ite)

                    fd = {
                        self.word_ids: word_ids,
                        self.true_labels[task_code]: label_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.current_learning_rate(),
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids,
                        self.language_flags[st.lang]: True
                    }

                    _, train_loss, gradient_norm = self.sess.run(
                        [self.train_op[task_code], self.loss[task_code], self.gradient_norm[task_code]]
                        , feed_dict=fd
                    )


                if st.task in ('dep'):
                    word_ids, sentence_lengths, char_ids, word_lengths, label_ids, arcs = next(ite)

                    fd = {
                        self.word_ids: word_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.current_learning_rate(),
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids,
                        self.arc_ids: arcs,
                        self.true_labels[task_code]: label_ids,
                        self.golden_arc_ids: True,
                        self.language_flags[st.lang]: True
                    }

                    self.sess.run([self.train_op[task_code]], feed_dict=fd)

                if st.task in ('nli'):
                    word_ids, sentence_lengths, char_ids, word_lengths, label_ids = next(ite)

                    fd = {
                        self.word_ids: word_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.current_learning_rate(),
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids,
                        self.true_labels[task_code]: label_ids,
                        self.language_flags[st.lang]: True
                    }

                    self.sess.run([self.train_op[task_code]], feed_dict=fd)

                if st.task in ('lmo'):
                    word_ids, sentence_lengths, char_ids, word_lengths = next(ite)
                    fd = {
                        self.word_ids: word_ids,
                        self.sequence_lengths: sentence_lengths,
                        self.learning_rate: self.current_learning_rate(),
                        self.dropout: self.config.dropout,
                        self.word_lengths: word_lengths,
                        self.char_ids: char_ids,
                        self.language_flags[st.lang]: True
                    }
                    self.sess.run([self.train_op[task_code]], feed_dict=fd)

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

            accs = []
            losses = []
            expected_ner = 0
            predicted_ner = 0
            precision = 0
            recall = 0

            for i, minibatch in enumerate(set_iterator):
                _, sentence_lengths, _, _,label_ids = minibatch

                labels_ids_predictions, loss = self.predict_crf_batch(minibatch, task_code, dev_set.lang)
                losses.append(loss)

                for lab, lab_pred, length in zip(label_ids, labels_ids_predictions, sentence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    for (true_t, pred_t) in zip(lab, lab_pred):
                        accs.append(true_t == pred_t)
                        if dev_set.task == 'ner':
                            O_token = self.dl.task_vocabs['ner'].token_to_id['O']
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
                precision = float(precision) / (predicted_ner + 1)
                recall = float(recall) / (expected_ner + 1)
                f1 = 2*precision*recall / (precision+recall + 1e-12)
                output.update({
                    'expected_ner_count': expected_ner,
                    'predicted_ner_count': predicted_ner,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        if dev_set.task in ('dep'):

            uases = 0
            lases = 0
            size = 0
            losses = []

            for i, minibatch in enumerate(set_iterator):

                word_ids, sentence_lengths, char_ids, word_lengths, label_ids, arcs = minibatch

                fd = {
                    self.word_ids: word_ids,
                    self.true_labels[task_code]: label_ids,
                    self.sequence_lengths: sentence_lengths,
                    self.dropout: 1,
                    self.word_lengths: word_lengths,
                    self.char_ids: char_ids,
                    self.arc_ids: arcs,
                    self.golden_arc_ids: False,
                    self.language_flags[dev_set.lang]: True
                }

                uas, las, loss = self.sess.run([self.uas, self.las, self.loss[task_code]], feed_dict=fd)
                uases += uas
                lases += las
                size += np.sum(sentence_lengths)
                losses.append(loss)

            output = {'uas': float(uases) / size, 'loss': np.mean(losses), 'las': float(lases) / size}

        if dev_set.task in ('nli'):

            counts = []
            losses = []
            sum = 0

            for i, minibatch in enumerate(set_iterator):
                word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch

                fd = {
                    self.word_ids: word_ids,
                    self.true_labels[task_code]: label_ids,
                    self.sequence_lengths: sentence_lengths,
                    self.dropout: 1,
                    self.word_lengths: word_lengths,
                    self.char_ids: char_ids,
                    self.language_flags[dev_set.lang]: True
                }

                count, loss = self.sess.run([self.correct_labels_count, self.loss[task_code]], feed_dict=fd)
                counts.append(count)
                losses.append(loss)
                sum += len(label_ids)

            output = {'acc': float(np.sum(counts)) / sum, 'loss': np.mean(losses)}

        if dev_set.task in ('lmo'):
            losses = []

            for i, minibatch in enumerate(set_iterator):
                word_ids, sentence_lengths, char_ids, word_lengths = minibatch

                fd = {
                    self.word_ids: word_ids,
                    self.sequence_lengths: sentence_lengths,
                    self.dropout: 1,
                    self.word_lengths: word_lengths,
                    self.char_ids: char_ids,
                    self.language_flags[dev_set.lang]: True
                }

                loss = self.sess.run([self.loss[task_code]], feed_dict=fd)
                losses.append(loss)
                # FIXME: perplexity

            output = {'loss': np.mean(losses)}

        return output

    def predict_crf_batch(self, minibatch, task_code, lang):

        word_ids, sentence_lengths, char_ids, word_lengths, label_ids = minibatch
        fd = {
            self.word_ids: word_ids,
            self.true_labels[task_code]: label_ids,
            self.sequence_lengths: sentence_lengths,
            self.dropout: 1,
            self.word_lengths: word_lengths,
            self.char_ids: char_ids,
            self.language_flags[lang]: True
        }

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params, loss = self.sess.run(
            [self.predicted_labels[task_code], self.trans_params[task_code], self.loss[task_code]], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sentence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, loss

