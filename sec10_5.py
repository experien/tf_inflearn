# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *
from tensorflow.examples.tutorials.mnist import input_data


# MNIST + softmax(dropout): 98.28%
# keep_prob: 학습 0.5~0.7, 테스트 1

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 15
num_iterations = int(mnist.train.num_examples / batch_size)

X, Y = ph([None, 784]), ph([None, 10])
keep_prob = ph()

W1 = varxavier("W1", [784, 512])
b1 = varnorm([512])
L1 = relu(mul(X, W1) + b1)
L1 = dropout(L1, keep_prob)

W2 = varxavier("W2", [512, 512])
b2 = varnorm([512])
L2 = relu(mul(L1, W2) + b2)
L2 = dropout(L2, keep_prob)

W3 = varxavier("W3", [512, 512])
b3 = varnorm([512])
L3 = relu(mul(L2, W3) + b3)
L3 = dropout(L3, keep_prob)

W4 = varxavier("W4", [512, 512])
b4 = varnorm([512])
L4 = relu(mul(L3, W4) + b4)
L4 = dropout(L4, keep_prob)

W5 = varxavier("W5", [512, 10])
b5 = varnorm([10])
hypo = mul(L4, W5) + b5

cost = rmean(cross_entropy(Y, hypo))
opt = opt_adam(learning_rate).minimize(cost)

open()
for epoch in range(num_epochs):
    avg_cost = 0
    for i in range(num_iterations):
        bx, by = mnist.train.next_batch(batch_size)
        c, _ = run([cost, opt], {X: bx, Y: by, keep_prob: 0.7})
        avg_cost += c / num_iterations

    print('Epoch:', epoch + 1, ', cost =', avg_cost)


pred = equal(argmax(hypo, 1), argmax(Y, 1))
acc = rmean(cast(pred, tfloat))
print('Accuracy:', run(acc, {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
