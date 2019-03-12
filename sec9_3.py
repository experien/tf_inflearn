# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# XOR using WIDE & DEEP Neural Net


x = np.array([[0, 0],   [0, 1],     [1, 0],     [1, 1]],    dtype=nfloat)
y = np.array([[0],      [1],        [1],        [0]],       dtype=nfloat)

X, Y = ph(), ph()

W1 = varnorm([2, 10], name='weight1')
b1 = varnorm([10], name='bias1')
layer1 = sigmoid(mul(X, W1) + b1)

W2 = varnorm([10, 10], name='weight2')
b2 = varnorm([10], name='bias2')
layer2 = sigmoid(mul(layer1, W2) + b2)

W3 = varnorm([10, 10], name='weight3')
b3 = varnorm([10], name='bias3')
layer3 = sigmoid(mul(layer2, W3) + b3)

W4 = varnorm([10, 1], name='weight4')
b4 = varnorm([1], name='bias4')
hypo = sigmoid(mul(layer3, W4) + b4)


cost = -rmean(Y * log(hypo) + (1 - Y) * log(1 - hypo))
opt = opt_grd(0.1).minimize(cost)


pred = cast(hypo > 0.5, tfloat)
acc = rmean(cast(equal(pred, Y), tfloat))


open()
for step in range(10001):
    run(opt, {X: x, Y: y})
    if step % 1000 == 0:
        print(step, run(cost, {X: x, Y: y}), run([W1, W2]))

h, p, a = run([hypo, pred, acc], {X: x, Y: y})
print('Hypothesis:', h, '\nPredicted:', p, '\nAccuracy:', a)

