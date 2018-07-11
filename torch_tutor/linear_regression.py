import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.linspace(-1, 1, 100).reshape(100, 1)  #torch 只会处理二维张量
y = x.pow(2) + 0.2 * torch.rand(x.shape)

x, y = Variable(x), Variable(y) #
# plt.scatter(x, y)
# plt.show()
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):#搭建曾需要的信息
        super(Net, self).__init__()  #官方步骤 先调用一些父类的init
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):#nn前向传递的过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1, 10, 1)
print(net)

plt.ioff()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    
    loss = loss_func (prediction, y)
    
    optimizer.zero_grad() # 类似于初始化，把梯度设为0
    loss.backward() #计算出梯度
    optimizer.step() #优化梯度
    
    ###画图过程
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss = %.4f' % loss.data[0], fontdict={'size':20, 'color': 'red'})
        plt.pause(0.1)
