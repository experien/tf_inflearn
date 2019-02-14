# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# DATA
x, y = [1, 2, 3, 4, 5], [2.1, 3.1, 4.1, 5.1, 6.1]

# HYPO
X, Y = ph(), ph()
W, b = varnorm([1]), varnorm([1])
hypo = X * W + b


# OPT
cost = rmean(tf.square(hypo - y))
opt = opt_grd(0.01).minimize(cost)


# TRAIN
open()
for step in range(2000):
    res = run([cost, W, b, opt], {X: x, Y: y})[:-1]
    if step % 100 == 0:
        print('#{}: cost={}, Y = X * {} + {}'.format(step, *res))


# TEST
print(run(hypo, {X: [5]}))
print(run(hypo, {X: [2.5]}))
print(run(hypo, {X: [1.5, 3.5]}))
