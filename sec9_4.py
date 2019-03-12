# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# XOR using WIDE & DEEP Neural Net + Tensorboard


x = np.array([[0, 0],   [0, 1],     [1, 0],     [1, 1]],    dtype=nfloat)
y = np.array([[0],      [1],        [1],        [0]],       dtype=nfloat)

X, Y = ph(), ph()

with tf.name_scope("Layer1") as scope:
    W1 = varnorm([2, 2], name='weight1')
    b1 = varnorm([2], name='bias1')
    layer1 = sigmoid(mul(X, W1) + b1)

    histo('Weight1', W1)
    histo('Bias1', b1)
    histo('Layer1', layer1)


with tf.name_scope("Layer2") as scope:
    W2 = varnorm([2, 1], name='weight2')
    b2 = varnorm([1], name='bias2')
    hypo = sigmoid(mul(layer1, W2) + b2)

    histo('Weight2', W2)
    histo('Bias2', b2)
    histo('Hypothesis', hypo)


with tf.name_scope("Cost"):
    cost = -rmean(Y * log(hypo) + (1 - Y) * log(1 - hypo))
    scalar('Cost', cost)

with tf.name_scope("Train"):
    opt = opt_grd(0.1).minimize(cost)


pred = cast(hypo > 0.5, tfloat)
acc = rmean(cast(equal(pred, Y), tfloat))
scalar('Accuracy', acc)

sess, _ = open()
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")
writer.add_graph(sess.graph)

for step in range(10001):
    s, _ = run([summary, opt], {X: x, Y: y})
    writer.add_summary(s, global_step=step)
    if step % 1000 == 0:
        print(step, run(cost, {X: x, Y: y}), run([W1, W2]))

h, p, a = run([hypo, pred, acc], {X: x, Y: y})
print('Hypothesis:', h, '\nPredicted:', p, '\nAccuracy:', a)

# $ tensorboard --logdir=./logs