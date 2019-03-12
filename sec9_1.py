# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# XOR using Ligistic regression


x = np.array([[0, 0],   [0, 1],     [1, 0],     [1, 1]],    dtype=nfloat)
y = np.array([[0],      [1],        [1],        [0]],       dtype=nfloat)

X, Y, W, b = ph(), ph(), varnorm([2, 1], name='weight'), varnorm([1], name='bias')
hypo = sigmoid(mul(X, W) + b)


cost = -rmean(Y * log(hypo) + (1 - Y) * log(1 - hypo))
opt = opt_grd(0.1).minimize(cost)


pred = cast(hypo > 0.5, tfloat)
acc = rmean(cast(equal(pred, Y), tfloat))


open()
for step in range(10001):
    run(opt, {X: x, Y: y})
    if step % 1000 == 0:
        print(step, run(cost, {X: x, Y: y}), run(W))

h, p, a = run([hypo, pred, acc], {X: x, Y: y})
print('Hypothesis:', h, '\nPredicted:', p, '\nAccuracy:', a)

