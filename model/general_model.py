import os

import tensorflow as tf
from datetime import datetime

from logs.logger_init import LoggerInit


class GeneralModel:

    def __init__(self, data_loader, config):
        self.config = config
        self.dl = data_loader
        self.name = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        self.logger = self.initialize_logger()

    type = tf.float32

    def add_train_op(self, loss, task_code):
        grads, vs = zip(*self.optimizer.compute_gradients(loss))
        self.gradient_norm[task_code] = tf.global_norm(grads)
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

    def build_graph(self):

        def add_hyperparameters(self):
            self.learning_rate = tf.placeholder(
                dtype=self.type, shape=[], name="lr"
            )

            self.dropout = tf.placeholder(
                dtype=self.type, shape=[], name="dropout"
            )

            # optimizer
            available_optimizers = {
                'rmsprop': tf.train.RMSPropOptimizer,
                'adagrad': tf.train.AdagradOptimizer,
                'adam': tf.train.AdamOptimizer,
                'sgd': tf.train.GradientDescentOptimizer
            }
            selected_optimizer = available_optimizers[self.config.optimizer]
            self.optimizer = selected_optimizer(self.learning_rate)

        add_hyperparameters(self)
        self._build_graph()

        config = None if self.config.use_gpu else tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        if self.config.show_graph:
            tf.summary.FileWriter(self.config.model_path, self.sess.graph)
            for variable in tf.global_variables():
                print(variable)

        self.saver = tf.train.Saver()

        if self.config.saved_model:
            self.saver.restore(self.sess, os.path.join(self.config.model_path, self.config.saved_model+".model"))
            reset_op = tf.group([v.initializer for v in self.optimizer.variables()])
            self.sess.run(reset_op)

    def close(self):
        # FIXME: Kill dataset threads (try to kill them even when Ctrl+C - or do they get killed with the shortcut as well?
        self.sess.close()
        tf.reset_default_graph()

    def initialize_logger(self):
        return LoggerInit(
            self.config.setup,
            filename=os.path.join(self.config.log_path, self.name),
            slack_channel=self.config.slack_channel,
            slack_token=self.config.slack_token
        ).logger
