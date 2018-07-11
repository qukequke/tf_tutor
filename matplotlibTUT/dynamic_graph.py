import matplotlib.pyplot as plt
import numpy as np


a = np.arange(16).reshape(4, 4)
# plt.ion()
for i in range(100):
    plt.imshow(a)
    a = np.fliplr(a)
    plt.draw()
    plt.pause(0.05)
# plt.ioff()
plt.show()
