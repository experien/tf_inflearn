# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# Training set, test set
# 데이터가 들쭉날쭉하거나 너무 클 때는 normalize한다. (예: MinMaxScaler())

x =      [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y =      [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
tx =     [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
ty =     [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X, Y = ph([None, 3]), ph([None, 3])
W, b = varnorm([3, 3]), [3]

hypo = softmax(mul(X, W) + b)
cost = rmean(-rsum(Y * log(hypo), axis=1))
opt = opt_grd(0.1).minimize(cost)

pred = argmax(hypo, 1)
is_correct = equal(pred, argmax(Y, 1))
acc = rmean(cast(is_correct, tfloat))

open()
for step in range(1000):
    cval, wval, _ = run([cost, W, opt], {X: x, Y: y})
    if step % 10 == 0:
        print(step, cval, wval)


print('Prediction:', run(pred, {X: x}))
print('Accuracy:', run(acc, {X: x, Y: y}))
