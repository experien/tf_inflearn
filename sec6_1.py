# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# Multinomial classification = Multinomial Logistic Rgression = Softmax Regression : 3개 이상 분류 문제.

# softmax: [2.0, 1.0, 0.1] => [0.7, 0.2, 0.1] 확률 => [1, 0, 0] one-hot encoding
# logit : softmax input
# cross-entropy cost function : logistic cost func.의 일반화

x = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

nb_classses = 3
X, Y = ph([None, 4]), ph([None, nb_classses])

W, b = varnorm([4, nb_classses]), varnorm([nb_classses])
# softmax = exp(logits) / rsum(exp(logits), dim)
hypo = softmax(mul(X, W) + b)

# cross entropy cost/loss
cost = rmean(-rsum(Y * log(hypo), axis=1))
opt = opt_grd(0.1).minimize(cost)

open()
for step in range(2001):
    run(opt, {X: x, Y: y})
    if step % 200 == 0:
        print(step, run(cost, {X: x, Y: y}))


h = run(hypo, {X: [[1, 11, 7, 9]]})
print(h, run(argmax(h, 1)))