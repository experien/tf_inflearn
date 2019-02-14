# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# 행렬


# Data
x = [[73., 80., 75.],
     [93., 88., 93.],
     [89., 91., 90.],
     [96., 98., 100.],
     [73., 66., 70.]]
y = [[152.], [185.], [180.], [196.], [142.]]


# HYPO
X, Y = ph([None, 3]), ph([None, 1]) # None -> undefined
W, b = varnorm([3, 1]), varnorm([1])
hypo = mul(X, W) + b


# OPT
cost = rmean(sqr(hypo - Y))
opt = opt_grd(1e-5).minimize(cost)


# TRAIN
open()
for step in range(2001):
    c, h, _ = run([cost, hypo, opt], {X: x, Y: y})
    if step % 100 == 0:
        print(f'#{step}: Cost={c}, Prediction=', *h)


# TEST
print('True Value =', *y)
