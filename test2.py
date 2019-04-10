import tensorflow as tf
import numpy as np

# NOTES:
# most machine learning models have parameters (weights/biases) and a cost function
# to evaluate accuracy of parameters. when training we're basically trying to
# find values of X so that f(x) (the cost function) is minimized <--> what this kinda
# means is that the prediction is closest to the target value

# gradient descent is a method that uses iteration -- starting with a set of parameters
# (weights and biases) it will improve them slowly to minimize the cost function


# neural network to learn the line of best fit given a bunch of points to predict the next point
# input data: 100 random points
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

# bias and weights
b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform([1, 2], -1, 1))

# constructing a linear model (basically is: y=mx+b)
y = tf.matmul(w, x_data) + b

# use gradient descent to improve our network over time -- its basically the 'machine learning' part
# mean squared error (reduced over time with training)
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # choose 0.5 as the learning rate
train = optimizer.minimize(loss)

# in tensorflow we wrap our computation in a graph
# initialize our tensorflow variables in our graph session
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# training process -- fit the plane
# in tensorflow you define your optimizer, create your session, and then run it
for step in range(0, 200):
    sess.run(train)

    # print the error every 20th step and see the value of the weight and bias
    # basically based on the data, it'll learn what the value of 'w' and 'b' should be
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))
