import tensorflow as tf
import numpy as np
import time 

# (x or y) and z

X = np.array([#1#2#3
              [0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])

y = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])

X_in = tf.placeholder(tf.float32,[3,1])
y_tar= tf.placeholder(tf.float32,[1,1])

W1= tf.Variable(tf.random_normal([4, 3]))
W2= tf.Variable(tf.random_normal([1, 4])) 

b1 = tf.Variable(tf.random_normal([1,1]))
b2 = tf.Variable(tf.random_normal([1,1]))
L1 = tf.nn.sigmoid(tf.add(tf.matmul(W1,X_in),b1))
L2 = tf.nn.sigmoid(tf.add(tf.matmul(W2,L1),b2))

err = tf.nn.sigmoid_cross_entropy_with_logits(logits=L2, labels=y_tar)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(err)

init_op = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init_op)

start = time.time()

for i in range(1,6000):
  sess.run(train,feed_dict={X_in:X[i%8,:].reshape((3,1)),y_tar:y[i%8,:].reshape((1,1))})

end = time.time()

print("training:",end-start)
start = time.time()

for i in range(0,8):
  print(sess.run(L2,feed_dict={X_in:X[i,:].reshape((3,1))}))

end = time.time()
print("evaluation:",end-start)

