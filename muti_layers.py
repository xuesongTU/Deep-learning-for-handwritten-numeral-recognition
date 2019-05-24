import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# number of neurones in each layer
K = 200
L = 100
M = 60
N = 30

# weights and bias
W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

#model
Y = tf.nn.softmac(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# placeholder for correct answers 
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 0.003 is the learning rate
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # load batch of images and correct answers 
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y: batch_Y}
    
    # train
    sess.run(train_step, feed_dict=train_step)
    
    # success ?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    
    # success on test data ?
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a, c = sess.run([accuracy, cross_entropy], feed=test_data)
