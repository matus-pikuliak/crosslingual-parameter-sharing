import os
import datetime

import tensorflow as tf

import utils.general_utils as utils
from logs.logger import Logger
from constants import LOG_CRITICAL, LOG_MESSAGE, LOG_RESULT


class GeneralModel:

    def __init__(self, data_loader, config):
        self.config = config
        self.dl = data_loader
        self.name = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        self.logger = self.initialize_logger()

    def build_graph(self):

        self.add_hyperparameters()
        self.add_optimizer()
        self._build_graph()

        config = None if self.config.use_gpu else tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.config.show_graph:
            self.show_graph()

        self.saver = tf.train.Saver(tf.trainable_variables())

        if self.config.load_model:
            self.load()

    def _build_graph(self):
        raise NotImplementedError

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def add_hyperparameters(self):
        self.learning_rate = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name="learning_rate")

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name="dropout")

    def add_optimizer(self):
        available_optimizers = {
            'rmsprop': tf.train.RMSPropOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'adam': tf.train.AdamOptimizer,
            'sgd': tf.train.GradientDescentOptimizer
        }
        selected_optimizer = available_optimizers[self.config.optimizer]
        self.optimizer = selected_optimizer(self.learning_rate)

    def add_train_op(self, loss):
        grads, vs = zip(*self.optimizer.compute_gradients(loss))
        if self.config.clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        return self.optimizer.apply_gradients(zip(grads, vs))

    def current_learning_rate(self):
        if self.config.learning_rate_schedule == 'static':
            return self.config.learning_rate
        elif self.config.learning_rate_schedule == 'decay':
            return self.config.learning_rate * pow(self.config.learning_rate_decay, self.epoch)
        else:
            raise AttributeError('lr_schedule must be set to static or decay')

    def run_experiment(self):
        start_time = datetime.datetime.now()
        self.log(f'Run started {start_time}', LOG_CRITICAL)
        self.log(f'{self.config}', LOG_MESSAGE)
        self.log(f'git hash: {utils.git_hash()}', LOG_MESSAGE)

        self.epoch = 1
        for i in range(self.config.epochs):
            self.run_epoch()

            if i == 0:
                epoch_time = datetime.datetime.now() - start_time
                self.log(f'ETA {start_time + epoch_time * self.config.epochs}', LOG_CRITICAL)

            if self.config.save_model == 'epoch':
                self.save(self.epoch)

            self.epoch += 1

        if self.config.save_model == 'run':
            self.save()

        end_time = datetime.datetime.now()
        self.log(f'Run done in {end_time - start_time}', LOG_CRITICAL)

    def run_epoch(self):
        raise NotImplementedError

    def show_graph(self):
        tf.summary.FileWriter(self.config.model_path, self.sess.graph)
        for variable in tf.global_variables():
            print(variable)
        exit()

    def save(self, global_step=None):
        self.saver.save(
            sess=self.sess,
            save_path=os.path.join(self.config.model_path, self.name),
            global_step=global_step,
            write_meta_graph=False)
        self.log(f'Model saved as {self.name}.', LOG_MESSAGE)

    def load(self):
        self.saver.restore(
            sess=self.sess,
            save_path=os.path.join(self.config.model_path, self.config.load_model))
        reset_op = tf.group([v.initializer for v in self.optimizer.variables()])
        self.sess.run(reset_op)
        self.log(f'Model restored from {self.config.load_model}.', LOG_MESSAGE)

    def log(self, message, level):
        self.logger.log(message, level)

    def initialize_logger(self):
        return Logger.factory(
            type=self.config.setup,
            server_name=self.config.server_name,
            filename=os.path.join(self.config.log_path, self.name),
            slack_channel=self.config.slack_channel,
            slack_token=self.config.slack_token)
