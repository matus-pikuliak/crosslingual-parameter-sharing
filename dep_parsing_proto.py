import tensorflow as tf


a = tf.placeholder(tf.float32, shape=[None, None, None]) # batch size x lenght x dim

root = tf.get_variable("root_vector", dtype=tf.float32, shape=[2]) # dim
root = tf.expand_dims(root, 0)
root = tf.expand_dims(root, 0)
root = tf.tile(
    root,
    [tf.shape(a)[0], 1, 1]
)
a_root = tf.concat([root, a], 1)

tile_a = tf.tile(
    tf.expand_dims(a, 2),
    [1, 1, tf.shape(a_root)[1], 1]
)
tile_b = tf.tile(
    tf.expand_dims(a_root, 1),
    [1, tf.shape(a)[1], 1, 1]
)
c = tf.concat([tile_a, tile_b], axis=3)

W = tf.get_variable("w", dtype=tf.float32, shape=[4, 20])
W2 = tf.get_variable("w2", dtype=tf.float32, shape=[20, 1])

c = tf.reshape(c, [-1, 4]) # 4 = dim * 2
d = tf.matmul(c, W)
d = tf.nn.relu(d)
d = tf.matmul(d, W2)
d = tf.reshape(d, [-1, 4]) # length+1 (root), toto nie je taka ista 4 ako +3 riadky hore
e = tf.nn.softmax(d)

tags_ph = tf.placeholder(tf.int32, shape=[2, 3]) # batch size x length
tags_oh = tf.one_hot(tags_ph, 4) # length+1

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=tags_oh,
    logits=d,
    dim=-1,
))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss=loss)

data = [
    [[1, 2],   [3,  4],  [4, 6]],
    [[11, 12], [13, 14], [3, 5]]
]

tags = [
    [2, 1, 0],
    [2, 3, 0]
]

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for _ in xrange(100000):
    _loss, _train_op = sess.run([loss, train_op], feed_dict={a: data, tags_ph: tags})
    print _loss

# print
# print res[1]


# sess.run(iter.initializer, feed_dict={ a: [4,5,6], b: [2,2,2]})
# for i in xrange(3):
#     res = sess.run([e])
#     print res

