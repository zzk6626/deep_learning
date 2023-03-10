import torch
import torch.utils.data as Data   # 小批训练的模块


BATCH_SIZE = 5      # 批训练的数据个数
                    # 如果BATCH_SIZE = 8 那么第一次训练输入8个数据，第二次训练输入2个数据


x = torch.linspace(1, 10, 10)       # x data (torch tensor)   [1 2 3 4 5 6 7 8 9 10]
y = torch.linspace(10, 1, 10)       # y data (torch tensor)   [10 9 8 7 6 5 4 3 2 1]

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x, y)
print(torch_dataset.__getitem__(5))  # 输出索引为第5（6）个的样本

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    # 每个epoch开始的时候，是否进行数据重排序，默认False
    shuffle=True,               # 要不要打乱数据 (打乱比较好),每次训练的数据不同，两个batchsize共10个，然后可以打乱每次训练的，但每个都会出现一次
    # num_workers=2,            ||  多线程来读数据
)

for epoch in range(3):   # 训练所有! 整套! 数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

"""
Epoch:  0 | Step:  0 | batch x:  [10.  6.  9.  2.  3.] | batch y:  [1. 5. 2. 9. 8.]
Epoch:  0 | Step:  1 | batch x:  [1. 7. 8. 5. 4.] | batch y:  [10.  4.  3.  6.  7.]
Epoch:  1 | Step:  0 | batch x:  [6. 8. 1. 5. 2.] | batch y:  [ 5.  3. 10.  6.  9.]
Epoch:  1 | Step:  1 | batch x:  [ 7.  4.  3. 10.  9.] | batch y:  [4. 7. 8. 1. 2.]
Epoch:  2 | Step:  0 | batch x:  [ 6.  1.  5. 10.  2.] | batch y:  [ 5. 10.  6.  1.  9.]
Epoch:  2 | Step:  1 | batch x:  [3. 7. 9. 8. 4.] | batch y:  [8. 4. 2. 3. 7.]
"""
