import os
import matplotlib.pyplot as plt
import numpy as np

IoU_paths = ['ultraslimS_1', 'ultraslimS_2', 'ultraslimS_3', 'ultraslimS_4']
fig = plt.figure()
for path in IoU_paths:
    config_path = os.path.join('/Users/nimariahi/Desktop',
    path)
    file_path = os.path.join(config_path, 'testIoU.txt')
    mynumbers = []
    x = []
    y = []
    with open(file_path, 'r')as f:
        for line in f:
            mynumbers.append([n for n in line.strip().split(' ')])
        for pair in mynumbers:
            try:
                x.append(pair[0])
                y.append(pair[1])
            except IndexError:
                print("A line in the file doesn't have enough entries.")
        plt.plot(np.asarray(x), np.asarray(y))
plt.title('Test data IoU after each Epoch')
plt.xlabel('Model')
plt.ylabel('IoU')
plt.legend(('configuration 1', 'configuration 2', 'configuration 3', 'configuration 4'))
plt.savefig(fig, format="png")
