# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *
from tensorflow.examples.tutorials.mnist import input_data


# 여러 기법들

# L2 regularization: # cost + lambda * sum(W**2)
# overfit을 막기 위해 weight 값이 커지면 cost가 커지도록(l2reg)

# Dropout

# Ensemble: 초기값만 다른 모델 n개를 독립적으로 훈련시킨 후 연결해서 예측

# NN 구성: fast foward, split&merge, cnn, rnn



# MNIST + softmax: 90%

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 15
num_iterations = int(mnist.train.num_examples / batch_size)


X, Y = ph([None, 784]), ph([None, 10])
W = varnorm([784, 10])
b = varnorm([10])
hypo = mul(X, W) + b

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
