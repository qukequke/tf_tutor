import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


a = torch.FloatTensor(np.arange(4).reshape(2,2))
torch.from_numpy(np.arange(4).reshape(2, 2))
a = torch.linspace(-2, 2, 100)
b = torch.pow(a, 2) + torch.rand(100)

# plt.scatter(a.numpy(), b.numpy())
# plt.show()
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn1 = torch.nn.Linear(1, 50)
        self.fcn2 = torch.nn.Linear(50, 1)
    def forward(self, x):
        x = self.fcn1(x)
        x = torch.relu(x)
        x = self.fcn2(x)
        return x
net = Net()
# net.parameters()

opti = torch.optim.Adam(net.parameters())
def loss_func(y, y_pre):
    return torch.mean(torch.pow((y - y_pre), 2))
a = a.reshape(100, 1)
b = b.reshape(100, 1)
# print(type(a))
# type(Variable(a))

steps = 1000
plt.ion()
for i in range(steps):
    x = Variable(a)
    y = Variable(b)
    pre = net(x)
    loss = loss_func(y, pre)
#     print(loss)
    opti.zero_grad()
    loss.backward()
    opti.step()
    if i % 5 == 0:
        # fig = plt.figure()
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pre.data.numpy(), c='r')
        plt.text(0.5, 0, 'loss = %.4f ' % loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
        plt.title('The ' + str(i) + 'step')
        plt.pause(0.2)
plt.ioff()
