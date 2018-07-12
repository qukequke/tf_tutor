import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import tensorflow as tf


c = torch.ones((100, 2))
x0 = torch.normal(2*c, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*c, 1)
y1 = torch.ones(100)
x = torch.cat([x0, x1], 0).type(torch.FloatTensor)
y = torch.cat([y0, y1], 0).type(torch.LongTensor)
print(y.shape)
x = Variable(x)
y = Variable(y)
print(x.type())
print(y.type())
# plt.scatter(x0.numpy()[:, 0], x0.numpy()[:, 1])
# plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1])
# plt.show()
net = torch.nn.Sequential(
    torch.nn.Linear(2, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 2)
    # torch.nn.Sigmoid()
)

optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.CrossEntropyLoss()
print(net)

# print(y.data.numpy())
# plt.figure()
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy())
# plt.show()

plt.ion()
for i in range(200):
    y_pre = net(x)
    # print(y_pre)
    # with tf.Session() as sess:
        # print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits=y_pre)))
    loss = loss_func(y_pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(y_pre)
    if i % 5 == 0:
        plt.cla()
        # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], cmap=y.data.numpy())
        y_ = torch.max(y_pre, 1)[1]
        acc = (sum(y_.data.numpy() == y.data.numpy())) / 200
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y_.data.numpy())
        plt.title(i)
        plt.text(2, -4, 'acc = %.2f %%' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
plt.ioff()


