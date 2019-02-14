# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# 행렬


# Data
x1, x2, x3 = [73., 93., 89., 96., 73.], [80., 88., 91., 98., 66.], [75., 93., 90., 100., 70.]
y = [152., 185., 180., 196., 142.]


# HYPO
X1, X2, X3, Y = ph(), ph(), ph(), ph()
W1, W2, W3, b = varnorm([1]), varnorm([1]), varnorm([1]), varnorm([1])
hypo = X1 * W1 + X2 + W2 + X3 * W3 + b


# OPT
cost = rmean(sqr(hypo - Y))
opt = opt_grd(1e-5).minimize(cost)


# TRAIN
open()
for step in range(10001):
    c, h, _ = run([cost, hypo, opt], {X1: x1, X2: x2, X3: x3, Y: y})
    if step % 1000 == 0:
        print(f'#{step}: Cost={c}, Prediction={h}')


# TEST