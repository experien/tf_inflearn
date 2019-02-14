# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# 여러 파일에서 읽어오기: Queue Runner


# Tensors
data = read_csvfiles(['data-01-test-score.csv', 'data-02-test-score.csv'], [[0.], [0.], [0.], [0.]])
BX, BY = batch([data[0:-1], data[-1:]], batch_size=10)


# HYPO
X, Y = ph([None, 3]), ph([None, 1]) # None -> undefined
W, b = varnorm([3, 1]), varnorm([1])
hypo = mul(X, W) + b


# OPT
cost = rmean(sqr(hypo - Y))
opt = opt_grd(1e-5).minimize(cost)


# TRAIN
open(queue_runner=True)
for step in range(2000):
    bx, by = run([BX, BY])
    c, h, _ = run([cost, hypo, opt], {X: bx, Y: by})
    if step % 100 == 0:
        print(f'#{step}: Cost = {c}, Prediction =', *h)


# TEST
print(*run(hypo, {X: [[100, 70, 101]]}))
print(*run(hypo, {X: [[60, 70, 110], [90, 100, 80]]}))
close()
