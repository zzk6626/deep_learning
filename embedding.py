import torch
embedding = torch.nn.Embedding(10, 3)
# A simple lookup table that stores embeddings of a fixed dictionary and size
# 一种简单的查找表，用于存储固定字典和大小的embedding

input = torch.LongTensor([1,2,3,4])
result = embedding(input) # shape: [4, 3]

print(result)
print(embedding.weight)

# 用反推的思想来看x1 * weight=result
# x1=[[0,1,0,0,0,0,0,0,0,0],
#  [0,0,1,0,0,0,0,0,0,0],
#  [0,0,0,1,0,0,0,0,0,0],
#  [0,0,0,0,1,0,0,0,0,0]]
# 这不就是one-hot编码嘛，只在自己所属的位置是1，其他位置都是0
# 还有一点，我们设置的num_embedding=10,这意味着我们input的输入值必须小于10，如果大于等于10就会报错。
# 总结
# embedding 管理着一个大小固定的二维向量权重，其input输入值它首先转化为one-hot编码格式，将转化为后的one-hot 与权重矩阵做矩阵乘法，就得到了每一个input的embedding输出。由于这个embedding权重是可训练的，所以在最训练后的权重值，能够表达不同字母之间的关系。
