# torch.util.data库学习 pytorch实现自由的数据读取
from torch.utils import data
import torch

data.Dataset()  # 创建数据集
data.Dataset.__getitem__(self=,index=) # 根据索引序号获取图片和标签，有__len__(self)函数来获取数据集的长度,其他数据集必须是data.Dataset的子类，比如说torchvision.ImageFolder

data.sampler.Sampler(data_source=)  # 创建一个采样器，该类是所有Sample的基类，其中，iter(self)函数来获取一个迭代器，对数据集中的元素的索引进行迭代，len(self)返回长度
data.sampler.Sampler.__iter__()

data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
# * dataset (Dataset): 加载数据的数据集
# * batch_size (int, optional): 每批加载多少个样本
# * shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False），仅打乱x，不打乱y
# * sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
# * batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
# * num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
# * collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
# * pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
# * drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
# * timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
# * worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．

# TensorDataset()的一个例子
class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# RandomSampler的一个例子
class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)

