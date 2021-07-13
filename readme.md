---
typora-root-url: E:\bwork\Learning\EasyNetwork
---

# Easy Network



## 什么是 Easy Network

Easy Network是一个不断收集各种神经网络模型的Git库，里面将以单个可执行.py文件形式保存每个网络模型的源代码。

由于深度学习入门较为困难，且很多项目代码过于复杂，因此学习周期较长。但实际上，绝大多数网络模型可以以单个py文件和仅仅几十行代码的形式就能实现。

因此本仓库为了简化学习过程，将每个网络模型单独保存为单独的一个可执行的.py文件，并包含一段测试网络输入与输出的代码以方便理解。



## 特点

在本仓库下的网络模型文件均满足以下三个基本条件：

1. 一个网路一个文件，一张结构图
2. 只依赖基本的数学库如：numpy，torch，torchvision
3. 可以直接运行，并输出数据尺寸与参数量

同时，每个网络模型还包含每一行的代码注释，以及特征图在传递过程中的尺寸注释。

最后，每个网络模型将包含一段被注释的代码，功能为使用tensorboardX绘制网络详细结构图，为了保证代码的易执行性，这段代码在每个文件中都将被注释掉，有需要的可以自行去掉#并运行。

```python
# from tensorboardX import SummaryWriter
# with SummaryWriter(comment='DenseNet') as w:
#   w.add_graph(net, inputs)
```



## 示例

CMD中运行 python alexnet.py 将得到以下输出：

```python
torch.Size([1, 1000]) params:61.101M (61100840)
按任意键结束
```

在文件夹中可查看AlexNet的网络结构图

![](/alexnet.png)