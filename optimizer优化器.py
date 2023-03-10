import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)    # 设置随机参数种子

# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# 构建数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))

# 绘制数据散点图
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 使用data loader
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 设置网络层network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# 分别对SGD和Monentum设置一个net
net_SGD         = Net()
net_Momentum    = Net()
nets = [net_SGD, net_Momentum]

# 设置不同的优化器，保证学习率相同，Momentum设置一个momentum动量参数
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)

optimizers = [opt_SGD, opt_Momentum]

loss_func = torch.nn.MSELoss()   # 回归的误差函数设置为MSE，均方误差
losses_his = [[], []]    # 记录网络训练时不同神经网络的 loss

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):

        # 对两个优化器, 优化属于响应神经网络
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())


labels = ['SGD','Momentum','RMSprop','Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his,label = labels[i])
plt.legend(loc='best')  # 图例即标签说明的位置
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))   # 设置y坐标的上下界限
plt.show()

