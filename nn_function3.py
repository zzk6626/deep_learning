import torch
import torch.nn as nn
import joblib

N, D_in, H, D_out = 64, 1000, 100, 10
# 随机创建训练数据
x = torch.randn(N, D_in)  # (64,1000)
y = torch.randn(N, D_out)   # (64,10)

class TwoLayerNet(torch.nn.Module):    # 从torch.nn.Module去继承，任何一个模型都是Module
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__() # 利用super的方法初始化类
        # define the model architecture
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayerNet(D_in, H, D_out)

# 有些模型适合 模型初始化优化 normal ，有些不适合
# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)

# model[0] 可以直接拿出来, model[0].bias/weight

# model = model.cuda()
loss_fn = nn.MSELoss(reduction='sum')  # 定义损失函数
# learning_rate = 1e-4  # Adam 一般用做1e-4  梯度反弹
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)  # 定义优化器
learning_rate = 1e-4  # SGD 一般用做1e-6, 效果较差
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)  # 定义优化器


for it in range(500):
    # forward pass
    y_pred = model(x)   #  与model.forward()一样的含义
    # compute loss
    loss = loss_fn(y_pred, y)   # computation graph
    print(it, loss.item())  # item()放下面，因为loss必须是tensor才能留下计算图

    # 保证在下一次求导之前更新为0就行，下一次backward之前
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # update weights of w1 and w2
    optimizer.step()

