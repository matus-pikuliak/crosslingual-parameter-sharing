import datetime

import numpy as np
import tensorflow as tf

hidden_size = 300
output_size = 15000
loss_type = 'softmax'
# loss_type = 'nce'
loss_type = 'sampled_softmax'
sampling = 10
steps = 100

hidden = tf.placeholder(
    dtype=tf.float32,
    shape=[None, hidden_size],)

word_logits_weights = tf.get_variable(
    name='word_logits_weights',
    shape=[hidden_size, output_size],
    dtype=tf.float32, )
word_logits_biases = tf.get_variable(
    name='word_logits_biases',
    shape=[output_size],
    dtype=tf.float32)

desired_word_ids = tf.placeholder(
    dtype=tf.int64,
    shape=[None])


optimizer = tf.train.AdamOptimizer(0.03)

logits = tf.nn.xw_plus_b(
    x=hidden,
    weights=word_logits_weights,
    biases=word_logits_biases)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=desired_word_ids,
    logits=logits)
loss = tf.reduce_mean(loss)
grads, vs = zip(*optimizer.compute_gradients(loss))
sm_train_op = optimizer.apply_gradients(zip(grads, vs))

loss = tf.nn.nce_loss(
    weights=tf.transpose(word_logits_weights),
    biases=word_logits_biases,
    labels=tf.expand_dims(desired_word_ids, axis=1),
    inputs=hidden,
    num_sampled=sampling,
    num_classes=output_size,
    partition_strategy='div')  # needed so the loss is consistent with softmax loss
loss = tf.reduce_mean(loss)
grads, vs = zip(*optimizer.compute_gradients(loss))
nce_train_op = optimizer.apply_gradients(zip(grads, vs))

loss = tf.nn.sampled_softmax_loss(
    weights=tf.transpose(word_logits_weights),
    biases=word_logits_biases,
    labels=tf.expand_dims(desired_word_ids, axis=1),
    inputs=hidden,
    num_sampled=sampling,
    num_classes=output_size,
    partition_strategy='div')  # needed so the loss is consistent with softmax loss
loss = tf.reduce_mean(loss)
grads, vs = zip(*optimizer.compute_gradients(loss))
samp_train_op = optimizer.apply_gradients(zip(grads, vs))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = datetime.datetime.now()
for _ in range(steps):
    sess.run(sm_train_op, feed_dict={
        hidden: np.random.random((32, hidden_size)),
        desired_word_ids: np.random.randint(0, output_size, size=(32))
    })
end_time = datetime.datetime.now()
print(f'Run done in {end_time - start_time}')

start_time = datetime.datetime.now()
for _ in range(steps):
    sess.run(nce_train_op, feed_dict={
        hidden: np.random.random((32, hidden_size)),
        desired_word_ids: np.random.randint(0, output_size, size=(32))
    })
end_time = datetime.datetime.now()
print(f'Run done in {end_time - start_time}')

start_time = datetime.datetime.now()
for _ in range(steps):
    sess.run(samp_train_op, feed_dict={
        hidden: np.random.random((32, hidden_size)),
        desired_word_ids: np.random.randint(0, output_size, size=(32))
    })
end_time = datetime.datetime.now()
print(f'Run done in {end_time - start_time}')
