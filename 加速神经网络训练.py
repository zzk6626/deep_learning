# data批训练可以一定程度上加快训练的效率
# 一般参数更新的过程  w = w - lr * w‘   摇摇晃晃走了很多弯路


# 让他处于滑坡上，由于向下的惯性，每次都会多走一点点，走的弯路就会变少
# MomentumGrad    w = w - lr * w‘   ||   m = b1 * m - lr * dx   ||    w = w + m

# 在学习率上做手脚，使得每一参数的更新都有它自己的方向，而是给他一双不好走路的鞋子，发现脚疼，鞋子变成走弯路的阻力，鞋子逼着他往前走
# AdaGrad         w = w - lr * w‘   ||   v += dx^2      ||  w = w - lr * dx / sqrt(v)

# 如果把下坡和不好走的鞋子结合起来
# RMSProp  = Momentum + AdaGrad  并未很好结合
# v = b1 * v + (1 - b1) * dx^2  || w = w - lr * dx / sqrt(v)


# Adam
# [momentum] m = b1 * m + (1 - b1) * dx  || [AdaGrad] v = b2 * v + (1 - b2) * dx^2   || w = w - lr * m / sqrt(v)

