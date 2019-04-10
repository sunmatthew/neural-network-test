import numpy as np

# NOTES:
# this is an example of a 3 layer feedforward neural network -- data comes in one way and out another

# sigmoid function: turns all of our input data into probabilities
def nonlin(x, deriv=False):
    if (deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

# input data
x = np.array([[0, 0, 1],
[0, 1, 1],
[1, 0, 1],
[1, 1, 1]])

# associated output data, each cell corresponds to a row in x
y = np.array([[0],
[1],
[1],
[0]])

# seed
np.random.seed(1)

# synapses -- two random matrices
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1


# training
# when we train our network, we're continously inputing that data and we're upating
# the weights over time (60,000 steps)
for j in range(60000):
    # layers
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # back propagation
    # reduces the error that the prediction is bad, the lower the error is the more
    # likely the prediction is correct
    l2_error = y - l2

    # prints this error everytime to show up how often during training
    # only print once every 10,000 steps
    if (j % 10000) == 0:
        print ('Error:' + str(np.mean(np.abs(l2_error))))

    # calculate deltas
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)


    # updating the weights (synapses)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ('output after training')
print (l2)
