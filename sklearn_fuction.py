import random
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn

iris = datasets.load_iris()
train_x = torch.Tensor(iris.data[0:1000])
train_y = torch.Tensor(iris.target[0:1000])

test_x = torch.Tensor(iris.data[101:105])
test_y = torch.Tensor(iris.data[101:105])

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

NUM_HIDDEN = 50

model = nn.Sequential(
    torch.nn.Linear(4, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 1)
)

if torch.cuda.is_available():
    model = model.cuda()

if torch.cuda.is_available():
    train_x = train_x.cuda()
    train_y = train_y.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for it in range(500):
    y_pred = model(train_x)
    loss = loss_fn(y_pred, train_y)
    acc = (y_pred == train_y)
    print(it, loss.item(),acc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
