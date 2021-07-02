
# %%
from ColorMNIST import ColorMNIST
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


batch_size = 32
path = '/home/andreas/praktikum/mnist'

# Load Data
train_data = ColorMNIST('both', 'train', path, randomcolor=True)
test_data = ColorMNIST('both', 'test', path, randomcolor=True)

#Define Train Loader and Test Loader
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()
images = images.numpy()
# %%


#Show 20 Samples from the batch
"""fig =  plt.figure(figsize=(20,5))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(images[idx].transpose([1,2,0]))
    ax.set_title(str(labels[idx].item()))

plt.show()"""

#Define the Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.fc1 = nn.Linear(int(28*28*64/4), 10)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, int(28*28*64/4))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        return x


#Train the Network
model = Net()
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.train()

n_epochs = 2
# %%
for epochs in range(n_epochs):
    train_loss = 0.0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

    train_loss = train_loss/len(train_loader.sampler)

    print(f'Epoch:{epochs +1}\t Training Loss{train_loss}')


# %%
