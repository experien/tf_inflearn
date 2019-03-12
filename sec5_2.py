# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# Logistic regression = binary classification
# 당뇨병 예측


# DATA
data = load_csv('data/data-03-diabetes.csv')
x, y = data[:, 0:-1], data[:, [-1]]


# HYPO
X, Y = ph([None, 8]), ph([None, 1])
W, b = varnorm([8, 1]), varnorm([1])
hypo = sigmoid(mul(X, W) + b)


# OPT
cost = -rmean(Y * log(hypo) + (1 - Y) * log(1 - hypo))
train = opt_grd(0.01).minimize(cost)


# ACCURACY
predicted = cast(hypo > 0.5, tfloat)
acc = rmean(cast(equal(predicted, Y), tfloat))


# TRAIN
open()
for step in range(10001):
    c, _ = run([cost, train], {X: x, Y: y})
    if step % 1000 == 0:
        print(step, c)


# REPORT
report = run([hypo, predicted, acc], {X: x, Y: y})
print("Hypothesis, Predicted, Accuracy = ", *report, sep='\n')
