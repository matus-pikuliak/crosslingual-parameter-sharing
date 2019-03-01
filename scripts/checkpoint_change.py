import sys

import tensorflow as tf


def change(checkpoint):

    change_list = {
        'dense/kernel': 'adversarial_training/dense/kernel',
        'dense/bias': 'adversarial_training/dense/bias',
        'dense_1/kernel': 'adversarial_training/dense_1/kernel',
        'dense_1/bias': 'adversarial_training/dense_1/bias',
    }

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint):
            print(var_name)
            var = tf.contrib.framework.load_variable(checkpoint, var_name)

            if var_name in change_list:
                var_name = change_list[var_name]

            var = tf.Variable(var, name=var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint)

change(*sys.argv[1:])