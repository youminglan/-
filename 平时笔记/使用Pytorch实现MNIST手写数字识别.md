### MNIST 数据集

​	mnist 数据集是一个非常出名的数据集，基本上很多网络都将其作为一个测试的标准，其来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST)。 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员，一共有 60000 张图片。 测试集(test set) 也是同样比例的手写数字数据，一共有 10000 张图片。



我们的任务就是给出一张图片，我们希望区别出其到底属于 0 到 9 这 10 个数字中的哪一个。



### 多分类问题

对于多分类问题而言，我们的 loss 函数使用一个更加复杂的函数，叫交叉熵。

提到交叉熵，我们先讲一下 softmax 函数，前面我们见过了 sigmoid 函数，如下

![image-20210413105713150](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413105713150.png)

可以将任何一个值转换到 0 ~ 1 之间，当然对于一个二分类问题，这样就足够了，因为对于二分类问题，如果不属于第一类，那么必定属于第二类，所以只需要用一个值来表示其属于其中一类概率，但是对于多分类问题，这样并不行，需要知道其属于每一类的概率，这个时候就需要 softmax 函数了。

交叉熵：

交叉熵衡量两个分布相似性的一种度量方式，前面讲的二分类问题的 loss 函数就是交叉熵的一种特殊情况，交叉熵的一般公式为：

![image-20210413105753074](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413105753074.png)



### 代码实现

```python
import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable

# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

a_data, a_label = train_set[0] 			#查看其中一个数据是什么样子
a_data = np.array(a_data, dtype='float32')
print(a_data.shape)						#图片大小是28x28

#对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据我们做一个变换，使用 reshape 将他们拉平成一个一维向量
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平成一个一维向量
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

a, a_label = train_set[0]
print(a.shape)			#torch.Size([784])
print(a_label)			#5

from torch.utils.data import DataLoader
# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
#使用这样的数据迭代器是非常有必要的，如果数据量太大，就无法一次将他们全部读入内存，所以需要使用 python 迭代器，每次生成一个批次的数据

a, a_label = next(iter(train_data))

# 打印出一个批次的数据大小
print(a.shape)
print(a_label.shape)

# 使用 Sequential 定义 4 层神经网络
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data), 
                     eval_loss / len(test_data), eval_acc / len(test_data)))
```

训练结果

![image-20210413110949851](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413110949851.png)

画出loss曲线和准确率曲线

``` python
import matplotlib.pyplot as plt
%matplotlib inline

plt.title('train loss')				#训练集loss
plt.plot(np.arange(len(losses)), losses)

plt.plot(np.arange(len(acces)), acces)#训练acc
plt.title('train acc')

plt.plot(np.arange(len(eval_losses)), eval_losses)#测试集loss
plt.title('test loss')

plt.plot(np.arange(len(eval_acces)), eval_acces)	#测试集acc
plt.title('test acc')
```

![image-20210413111235706](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413111235706.png)

![image-20210413111250259](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413111250259.png)

![image-20210413111301306](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413111301306.png)

![image-20210413111317570](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413111317570.png)

可以看到我们的三层网络在训练集上能够达到 99.9% 的准确率，测试集上能够达到 98.20% 的准确率