# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# Fancy softmax classifier: Animal classification

data = load_csv('data/data-04-zoo.csv')
x, y = data[:, 0:-1], data[:,[-1]]


nb_classes = 7
X, Y = ph([None, 16]), ph([None, 1], dt=tint) # 0~6, shpae=(?, 1)

Y_one_hot = one_hot(Y, nb_classes) # one-hot shape=(?, 1, 7) 한 차원이 더해짐
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape=(?, 7)

W, b = varnorm([16, nb_classes]), varnorm([nb_classes])

logits = mul(X, W) + b
hypo = softmax(logits)

cost_i = cross_entropy(Y_one_hot, logits)
cost = rmean(cost_i)
opt = opt_grd(0.1).minimize(cost)

predicted = argmax(hypo, 1)
correct = argmax(Y_one_hot, 1)
acc = rmean(cast(equal(predicted, correct), tfloat))

open()
for step in range(2000):
    run(opt, {X: x, Y: y})
    if step % 100 == 0:
        c, a = run([cost, acc], {X: x, Y: y})
        print(step, c, a)

p = run(predicted, {X: x})
for p, y in zip(p, y.flatten()):
    print("[{}] prediction: {} true value: {}".format(p == int(y), p, int(y)))
