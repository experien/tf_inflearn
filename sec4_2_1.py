# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# file


# Data
data = load_csv('data-01-test-score.csv')
x, y = data[:,:-1], data[:,-1:]
# print(x)
# print(y)


# HYPO
X, Y = ph([None, 3]), ph([None, 1]) # None -> undefined
W, b = varnorm([3, 1]), varnorm([1])
hypo = mul(X, W) + b


# OPT
cost = rmean(sqr(hypo - Y))
opt = opt_grd(1e-5).minimize(cost)


# TRAIN
open()
for step in range(10000):
    c, h, _ = run([cost, hypo, opt], {X: x, Y: y})
    if step % 1000 == 0:
        print(f'#{step}: Cost={c}, Prediction=', *h)


# TEST
print(*run(hypo, {X: [[100, 70, 101]]}))
print(*run(hypo, {X: [[60, 70, 110], [90, 100, 80]]}))
