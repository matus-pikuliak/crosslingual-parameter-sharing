import re

import tensorflow as tf


def dense_with_weights(inputs, units, activation):

    hidden = tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation)

    scope_name = hidden.name.split('/')[1]
    with tf.variable_scope(scope_name, reuse=True):
        kernel = tf.get_variable('kernel')

    return hidden, kernel
