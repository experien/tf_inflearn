# https://github.com/hunkim/DeepLearningZeroToAll/
from tensorflow_wrapper import *


# cost func. -> 경사하강법 직접 구현 / optimizer 사용


# DATA
x, y = [1, 2, 3], [1, 2, 3]


# HYPO
X, Y = ph(), ph()
W = varnorm([1])
#W = var(5.)
hypo = X * W


# OPT
cost = rmean(sqr(hypo - Y))
# learning_rate = 0.1
# gradient = rmean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)
opt = opt_grd(0.1).minimize(cost)

# TRAIN
open()
Wlst, clst = [], []
for i in range(21):
    #w = run(update, {X: x, Y: y})
    w, c, _ = run([hypo, W, cost, opt], {X: x, Y: y})
    Wlst.append(w)
    clst.append(c)


# TEST
plt.plot(range(21), Wlst, label="W")
plt.plot(range(21), clst, label="cost")
plt.legend()
plt.xlabel("step")
plt.show()
