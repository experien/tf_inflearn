# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *
from tensorflow.examples.tutorials.mnist import input_data
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
X, Y = ph([None, 784]), ph([None, nb_classes])
W, b = varnorm([784, nb_classes]), varnorm([nb_classes])

hypo = softmax(mul(X, W) + b)
cost = rmean(-rsum(Y * log(hypo), axis=1))
opt = opt_grd(0.1).minimize(cost)

is_correct = equal(argmax(hypo, 1), argmax(Y, 1)),
acc = rmean(cast(is_correct, tfloat))

epochs = 15
batch_size = 100
open()
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        bx, by = mnist.train.next_batch(batch_size)
        c, _ = run([cost, opt], {X: bx, Y: by})
        avg_cost += c / total_batch

    print('Epoch#{:04d} cost = {:.9f}'.format(epoch + 1, avg_cost))

#print('Accuracy:', acc.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print('Accuracy:', run([acc], {X: mnist.test.images, Y: mnist.test.labels}))


r = random.randint(0, mnist.test.num_examples - 1)
test_img = mnist.test.images[r:r+1]
test_label = mnist.test.labels[r:r+1]

print("Label:", run(argmax(test_label, 1)))
print("Prediction:", run(argmax(hypo, 1), {X: test_img}))
plt.imshow(test_img.reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()