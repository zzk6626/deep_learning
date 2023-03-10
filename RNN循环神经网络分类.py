import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height / 28次，
INPUT_SIZE = 28         # rnn input size / image width  / 一行28个
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data


# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,  # 一次输入28个像素点（一行）
            hidden_size=64,         # LSTM中隐层的维度
            num_layers=1,           # 循环神经网络的层数 ，数值越大，LSTM/RNN能力越强 但是拖的时间越长
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size) 该段代码会交换维度位置为(time_step, batch,  input_size)
            # dropout　默认是0，代表不用dropout
            # bidirectional默认是false，代表不用双向LSTM
        )
            # 注意LSTM中的隐藏状态其实就是输出
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])   # (batch, time_step, input_size) 选取最后一个hidden state
        return out


rnn = RNN()
rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func.cuda()

start_time = time.time()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data

        b_x = b_x.cuda()
        b_y = b_y.cuda()

        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss

        print(step, loss.item())  # item()放下面，因为loss必须是tensor才能留下计算图

        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

    # 测试步骤开始
    rnn.eval()   # 这时候我们要找一下Model里是否有BN或者 Dropout层，如果存在了，那就要小心了！！
    test_x = test_x.cuda()
    test_y = test_y.cuda()
    output = rnn(test_x)
    acc = (output.argmax(1) == test_y).sum()

print(acc)
end_time = time.time()
print(end_time - start_time)
# print 10 predictions from test data

