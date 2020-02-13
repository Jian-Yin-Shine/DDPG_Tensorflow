'''
import tensorflow as tf

x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4, 1))

with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
dy_dx = g.gradient(y, x)
dz_dx = g.gradient(z, x)

print(dy_dx, dz_dx)
'''


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal((num_examples, num_inputs), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=1)

# print(features[0], labels[0])

def data_iter(batch_size, features, labels):
    features = np.array(features)
    labels = np.array(labels)
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i+batch_size, num_examples)])
        yield features[j], labels[j]

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=1))
b = tf.Variable(tf.zeros((1,)))

# print(w, b)

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 /2

def sgd(params, lr, batch_size, grads):
    for i, params in enumerate(params):
        params.assign_sub(lr * grads[i] / batch_size)

lr = 0.03
num_epochs = 100
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as t:
            t.watch([w,b])
            l = loss(net(X, w, b), y)
        grads = t.gradient(l, [w, b])
        sgd([w, b], lr, batch_size, grads)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))

print(w)
print(b)