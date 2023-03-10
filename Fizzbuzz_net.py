import torch
import numpy as np
from Fizzbuzz import fizz_buzz_encode, fizz_buzz_decode

NUM_DIGITS = 10  # 二进制位数限制位数

# 已知此网络不易训练，改换成二进制的数字容易学习
# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])
    # >> 右移运算符 高位补0

trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# 建立模型
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4) # 4 logits, after softmax, we get probability distribution
)

if torch.cuda.is_available():
    model = model.cuda()

# 分类问题一般用 交叉熵损失函数
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):   # 等价于[0:len(trX):128]
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        print("Epoch", epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()  # backward
        optimizer.step()

testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1,101)])
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():
    testY = model(testX)

predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))
print([fizz_buzz_decode(i, x) for (i, x) in predictions])

print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])))
testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])
