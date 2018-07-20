import tensorflow as tf

lstm_size = 1
tf_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None]) # batch_size x length x dim
sequence_lengths = tf.placeholder(dtype=tf.float32, shape=[None]) # batch_size x length x dim


root = tf.get_variable("root_vector", dtype=tf.float32, shape=[2 * lstm_size])  # dim
root = tf.expand_dims(root, 0)
root = tf.expand_dims(root, 0)
root = tf.tile(
    root,
    [tf.shape(tf_input)[0], 1, 1]
)
a_root = tf.concat([root, tf_input], 1)

tile_a = tf.tile(
    tf.expand_dims(tf_input, 2),
    [1, 1, tf.shape(a_root)[1], 1]
)
tile_b = tf.tile(
    tf.expand_dims(a_root, 1),
    [1, tf.shape(tf_input)[1], 1, 1]
)

c = tf.concat([tile_a,
               tile_b], axis=3)

W = tf.get_variable("w", dtype=tf.float32, shape=[4 * lstm_size, 200])
b = tf.get_variable("b", dtype=tf.float32, shape=[200])
W2 = tf.get_variable("w2", dtype=tf.float32, shape=[200, 1])

c = tf.reshape(c, [-1, 4 * lstm_size])
d = tf.matmul(c, W) + b
d = tf.nn.relu(d)
d = tf.matmul(d, W2)
d = tf.reshape(d, [-1, tf.shape(a_root)[1]])  # length+1 (root)

seq_mask = tf.reshape(tf.sequence_mask(sequence_lengths), shape=[-1])

e = tf.boolean_mask(d, seq_mask)
#
tags_ph = tf.placeholder(tf.int64, shape=[None, None])  # batch size x length
tags_oh = tf.one_hot(tags_ph, tf.shape(a_root)[1])  # length+1

d2 = tf.argmax(d, axis=1)
d3 = tf.reshape(tags_ph, shape=[-1])
pred_diff = tf.count_nonzero(d2 - d3)
uad = 1 - tf.cast(pred_diff, tf.float32) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)

oink = tf.nn.softmax_cross_entropy_with_logits(
    labels=tags_oh,
    logits=d,
    dim=-1,
)

seq_mask = tf.reshape(tf.sequence_mask(sequence_lengths), shape=[-1])

oinker = tf.boolean_mask(oink, seq_mask)

dep_loss = tf.reduce_mean(oinker)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
dep_train_op = optimizer.minimize(loss=dep_loss)

data = [
    [[1, 2],   [3,  4],  [4, 6]],
    [[11, 12], [13, 14], [0, 0]]
]

tags = [
    [2, 1, 0],
    [1, 0, 0]
]

lens = [
    2, 3
]

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print sess.run([d, e], feed_dict={tf_input: data, tags_ph: tags, sequence_lengths: lens})
exit()

for _ in xrange(100000):
    _, ed, eo = sess.run([dep_train_op, oink, oinker], feed_dict={tf_input: data, tags_ph: tags, sequence_lengths: lens})
    print ed, eo

# print
# print res[1]


# sess.run(iter.initializer, feed_dict={ a: [4,5,6], b: [2,2,2]})
# for i in xrange(3):
#     res = sess.run([e])
#     print res

