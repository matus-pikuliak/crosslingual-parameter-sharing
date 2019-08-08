import os
import datetime
import types
from functools import reduce

import numpy as np
import tensorflow as tf

import utils.general_utils as utils
from logs.logger import Logger
from constants import LOG_CRITICAL, LOG_MESSAGE, LOG_RESULT
from model.model import Model


class Orchestrator:

    def __init__(self, data_loader, config, name=None):
        self.config = config
        self.dl = data_loader

        if not name:
            self.name = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        else:
            self.name = name

        self.logger = Logger.factory(
            type=self.config.setup,
            server_name=self.config.server_name,
            filename=os.path.join(self.config.log_path, self.name),
            slack_channel=self.config.slack_channel,
            slack_token=self.config.slack_token)

        self.n = types.SimpleNamespace()

        self.tasks, self.langs = zip(*self.config.tasks)

    def __enter__(self, *args):
        self.add_hyperparameters()
        self.add_optimizer()

        self.models = {tl: Model.factory(*tl, self, self.dl, self.config, self.logger) for tl in self.config.tasks}
        for model in self.models.values():
            model.build_graph()

        config = tf.ConfigProto()
        if not self.config.use_gpu:
            config.device_count = {'GPU': 0, 'CPU': 1}
        if self.config.allow_gpu_growth:
            config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.config.show_graph:
            self.show_graph()

        self.saver = tf.train.Saver(
            var_list=tf.trainable_variables(),
            max_to_keep=None)

        if self.config.load_model:
            self.load()
        return self

    def __exit__(self, *exc_info):
        self.sess.close()
        tf.reset_default_graph()

    def add_hyperparameters(self):
        self.n.learning_rate = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name="learning_rate")

        self.n.dropout = tf.placeholder(
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
        self.n.optimizer = selected_optimizer(self.n.learning_rate)

    def current_learning_rate(self, epoch=None):
        if epoch is None:
            epoch = self.epoch

        if self.config.learning_rate_schedule == 'static':
            return self.config.learning_rate
        elif self.config.learning_rate_schedule == 'decay':
            return self.config.learning_rate * pow(self.config.learning_rate_decay, epoch-1)
        else:
            raise AttributeError('lr_schedule must be set to static or decay')

    def run_training(self, start_epoch=1):
        if self.config.early_stopping > 0:
            try:
                focused = next(model for model in self.models.values() if model.is_focused())
            except StopIteration:  # no model is focused
                assert len(self.models) == 1
                focused = list(self.models.values())[0]

        start_time = datetime.datetime.now()
        self.log(
            message=f'Run started {start_time}',
            level=LOG_CRITICAL)
        self.log(
            message={
                'config': self.config,
                'git_hash': utils.git_hash()
            },
            level=LOG_RESULT)

        for self.epoch in range(start_epoch, self.config.epochs+1):
            self.run_epoch()

            if self.epoch == 1:
                epoch_time = datetime.datetime.now() - start_time
                self.log(
                    message=f'ETA {start_time + epoch_time * self.config.epochs}',
                    level=LOG_CRITICAL)

            if self.config.save_model == 'epoch':
                self.save(self.epoch)

            if self.config.early_stopping > 0 and focused.should_early_stop():
                self.log(
                    message=f'Early stopping in epoch {self.epoch}',
                    level=LOG_CRITICAL)
                break

        if self.config.save_model == 'run':
            self.save()

        end_time = datetime.datetime.now()
        self.log(
            message=f'Run done in {end_time - start_time}',
            level=LOG_CRITICAL)

    def run_epoch(self):

        train_models = [m for m in self.models.values()]
        if self.config.train_only is not None:
            train_models = [m for m in train_models if [m.task, m.lang] == self.config.train_only.split('-')]

        if self.config.focus_on is None:
            off_rate = 1 / len(train_models)
        else:
            on_rate = self.config.focus_rate
            off_rate = (1 - self.config.focus_rate) / (len(train_models) - 1)

        probs = {
            model: on_rate if model.is_focused() else off_rate
            for model
            in train_models
        }

        for _ in range(self.config.epoch_steps):
            sampled_model = np.random.choice(list(probs.keys()), p=list(probs.values()))
            sampled_model.train_step()

        self.log(f'Epoch {self.epoch} training done.', LOG_MESSAGE)

        self.evaluate_models()

    def run_evaluation(self, start_epoch=1):
        """
        This method runs only evaluation phase on pretrained models
        """
        start_time = datetime.datetime.now()
        self.log(
            message=f'Run started {start_time}',
            level=LOG_CRITICAL)
        self.log(
            message={
                'config': self.config,
                'git_hash': utils.git_hash()
            },
            level=LOG_RESULT)

        for self.epoch in range(start_epoch, self.config.epochs+1):
            self.load(f'{self.name}-{self.epoch}')
            self.evaluate_models()

        end_time = datetime.datetime.now()
        self.log(
            message=f'Run done in {end_time - start_time}',
            level=LOG_CRITICAL)

    def evaluate_models(self):
        eval_models = self.models.values()

        if self.config.focus_on is not None:
            eval_models = [self.models[tuple(self.config.focus_on.split('-'))]]

        if self.config.train_only is not None:
            eval_models = [self.models[tuple(self.config.train_only.split('-'))]]

        for model in eval_models:
            model.evaluate()

    def show_graph(self):
        tf.summary.FileWriter(self.config.model_path, self.sess.graph)
        size = 0
        for variable in tf.global_variables():
            if 'Adam' not in variable.name:
                print(variable)
            size += reduce(lambda x, y: x*y, variable.shape, 1)
        print(f'Total size: {size}.')

    def save(self, global_step=None):
        self.saver.save(
            sess=self.sess,
            save_path=os.path.join(self.config.model_path, self.name),
            global_step=global_step,
            write_meta_graph=False)
        self.log(f'Model saved as {self.name}.', LOG_MESSAGE)

    def load(self, model_file=None):
        if model_file is None:
            model_file = self.config.load_model

        self.saver.restore(
            sess=self.sess,
            save_path=os.path.join(self.config.model_path, model_file))
        reset_op = tf.group([v.initializer for v in self.n.optimizer.variables()])
        self.sess.run(reset_op)
        self.log(f'Model restored from {model_file}.', LOG_MESSAGE)

    def log(self, message, level):
        self.logger.log(message, level)
