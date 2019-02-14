# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# cost func. 그려보기


# DATA
X, Y = [1, 2, 3], [1, 2, 3]


# HYPO
W = ph()
hypo = X * W


# OPT
cost = rmean(sqr(hypo - Y))


# TRAIN
open()
Wlst, costlst = [], []
for i in range(-30, 50):
    c, w = run([cost, W], {W: i * 0.1})  # W = -3~5
    Wlst.append(w)
    costlst.append(c)


# TEST
plt.plot(Wlst, costlst)
plt.show()
