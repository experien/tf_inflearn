# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# Logistic regression = binary classification
# logistic function = sigmoid function ? 0 ~ 1 사이 값

# H(X) = sigmoid(HX)
# cost => 쭈글쭈글해짐(non-convex)
# y = 0 또는 1 이라는 점, + log를 사용해서
# cost 함수를 convex 로 만듦 -> 경사하강법 사용해서 학습 가능


# DATA
x = [[1, 2], [2, 3], [3,1], [4, 3], [5, 3], [6, 2]]
y = [[0], [0], [0], [1], [1], [1]] # 1(pass) 또는 0(fail)



# HYPO
X, Y = ph([None, 2]), ph([None, 1])
W, b = varnorm([2, 1]), varnorm([1])
hypo = sigmoid(mul(X, W) + b) # 0~1 사이


# OPT
# Y=0: hypo=0 일 때(예측 맞음) cost=0, hypo=1 일 때(예측 틀림) cost=Inf
# Y=1: hypo=0 일 때(예측 틀림) cost=Inf, hypo=1 일 때(예측 맞음) cost=0
cost = -rmean(Y * log(hypo) + (1 - Y) * log(1 - hypo))
train = opt_grd(0.01).minimize(cost)


# ACCURACY
predicted = cast(hypo > 0.5, tfloat)
acc = rmean(cast(equal(predicted, Y), tfloat))


# TRAIN
open()
for step in range(10001):
    c, _ = run([cost, train], {X: x, Y: y})
    if step % 500 == 0:
        print(step, c)


# REPORT
print("Hypothesis, Predicted, Accuracy = ", *run([hypo, predicted, acc], {X: x, Y: y}), sep='\n')
