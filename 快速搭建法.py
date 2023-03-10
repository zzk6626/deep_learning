import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型 0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型 0 y data (tensor), shape=(100, )    标签 0
x1 = torch.normal(-2*n_data, 1)     # 类型 1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # 类型 1 y data (tensor), shape=(100, )    标签 1
# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
print(net2)

'''
Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
)
'''
# 自己定义的类无需实例化
class Net(torch.nn.Module):     # 继承 torch 的 Module   2 -> 10 -> 2  ||  二分类的例子 最后输出为2  [0 1] [1 0]
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)  激活函数relu
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net1 = Net(n_feature=2, n_hidden=10, n_output=2)   # 定义好网络必须直接输入，但是正则化需要先进行 实例化
print(net1)
'''
Net(
  (hidden): Linear(in_features=2, out_features=10, bias=True)
  (out): Linear(in_features=10, out_features=2, bias=True)
)
# F.relu相当于一个功能，因此不会显示层结构里面
'''