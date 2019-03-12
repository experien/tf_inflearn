# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *
from tensorflow.examples.tutorials.mnist import input_data


# MNIST + softmax(xavier init): 97.8%

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 15
num_iterations = int(mnist.train.num_examples / batch_size)


X, Y = ph([None, 784]), ph([None, 10])

W1 = varxavier("W1", [784, 256])
b1 = varnorm([256])
L1 = relu(mul(X, W1) + b1)

W2 = varxavier("W2", [256, 256])
b2 = varnorm([256])
L2 = relu(mul(L1, W2) + b2)

W3 = varxavier("W3", [256, 10])
b3 = varnorm([10])
hypo = mul(L2, W3) + b3

cost = rmean(cross_entropy(Y, hypo))
opt = opt_adam(learning_rate).minimize(cost)

open()
for epoch in range(num_epochs):
    avg_cost = 0
    for i in range(num_iterations):
        bx, by = mnist.train.next_batch(batch_size)
        c, _ = run([cost, opt], {X: bx, Y: by})
        avg_cost += c / num_iterations

    print('Epoch:', epoch + 1, ', cost =', avg_cost)


pred = equal(argmax(hypo, 1), argmax(Y, 1))
acc = rmean(cast(pred, tfloat))
print('Accuracy:', run(acc, {X: mnist.test.images, Y: mnist.test.labels}))
