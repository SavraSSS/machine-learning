import torch as th
import torchvision.datasets
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)
th.backends.cudnn.deterministic=True


MNIST_train = torchvision.datasets.MNIST('./', download = True, train = True)
MNIST_test = torchvision.datasets.MNIST('./', download = True, train = False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train.dtype, y_train.dtype

X_train = X_train.float()
X_test = X_test.float()

X_train.shape, X_test.shape

plt.imshow(X_train[8, :, :])
plt.show()
print(y_train[8])

plt.imshow(X_train[19, :, :])
plt.show()
print(y_train[19])

plt.imshow(X_train[1, :, :])
plt.show()
print(y_train[1])

X_train = X_train.reshape([-1, 28*28])
X_test = X_test.reshape([-1, 28*28])

print(X_train)

count_col = 28*28

class MNISTnet(th.nn.Module):
  def __init__(self, n):
    super (MNISTnet, self).__init__()
    self.fc1 = th.nn.Linear(count_col, n)
    self.act1 = th.nn.ReLU()
    self.fc2 = th.nn.Linear(n, n)
    self.act2 = th.nn.Tanh()
    self.out = th.nn.Linear(n, 10)
    self.sm = th.nn.Softmax(dim=1)


  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.out(x)
    return x




  def predict(self, data_input):
    with th.no_grad():
      x = self.forward(data_input)
      x = x.reshape(1,10)
      x = self.sm(x)
      _, predicted = th.max(x, 1)
      return predicted.item()
net = MNISTnet(500)

th.cuda.is_available()

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
net = net.to(device)

loss = th.nn.CrossEntropyLoss()
opt = th.optim.Adam(net.parameters(), lr = 0.001)

batch_size = 50
epochs = 40
train_losses = []
test_losses = []
X_test = X_test.to(device)
y_test = y_test.to(device)
for epoch in range(epochs):
  order = np.random.permutation(len(X_train))
  for start_index in range (0,len(X_train), batch_size):
    opt.zero_grad()
    batch_index = order[start_index:start_index+batch_size]
    X_batch = X_train[batch_index]
    y_batch = y_train[batch_index]
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    preds = net.forward(X_batch)
    loss_val = loss(preds, y_batch)
    loss_val.backward()
    opt.step()
  train_loss = loss_val.item()
  train_losses.append(train_loss)
  test_preds = net.forward(X_test)
  test_loss = loss(test_preds, y_test)
  test_loss_value = test_loss.item()
  test_losses.append(test_loss_value)

  test_preds = test_preds.argmax(dim=1)
  print((test_preds==y_test).float().mean())

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

X_train = X_train.to(device)
print(net.predict(X_train[8, :]))
print(net.predict(X_train[19, :]))
print(net.predict(X_train[1, :]))