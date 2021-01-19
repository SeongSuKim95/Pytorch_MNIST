import torch
from torch import nn
from torch import optim
## Datasets
from torchvision import datasets, transforms
## Data Split, Data Loader
from torch.utils.data import random_split , DataLoader
import cv2

#torch.randn(5).cuda()

## Define model
model = nn.Sequential(
    nn.Linear(28*28,64), ## First Hidden layer ->64
    nn.ReLU(),
    nn.Linear(64,64),## Second Hidden layer -> 64
    nn.ReLU(),
    nn.Linear(64,10) ## Output layer
                     )
## Define optimizer
params = model.parameters()
optimizer = optim.SGD(params, lr = 1e-2)
model = model.cuda()

## Define loss

loss = nn.CrossEntropyLoss()

## Train, Val split

train_set = datasets.MNIST('mnist',train=True, download = True, transform = transforms.ToTensor())

#Visualizing MNIST
print(train_set.train_data.shape)
mnist_test = train_set.train_data[0].numpy()
print(mnist_test)
print(mnist_test.size)
cv2.imshow('mnist',mnist_test)
cv2.waitKey(0)
###

train, val = random_split(train_set,[55000,5000])
train_loader = DataLoader(train, batch_size = 32)
val_loader = DataLoader(val, batch_size = 32)

# My training and validation loops
Epochs = 5
for epoch in range(Epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch
        #x : b x 1 x 28 x 28
        b = x.size(0)
        #x = x.view(b,-1)
        x = x.view(b,-1).cuda()
        # Step 1 -> Forward
        l = model(x) # l: logits

        # Step 2 -> Compute the objective function
        #J = loss(l,y) # y = labels
        J = loss(l,y.cuda())
        # Step 3 -> Cleaning the gradients
        model.zero_grad()
        # == optimizer.zero_grad()
        # == params.grad._zero()

        # Step 4 -> Accumulate the partial derivatives of J wrt params
        J.backward()
        # parms.grad._sum(dJ/dparams)

        # Step 5 -> Opposite direction of the gradient
        optimizer.step()
        # with torch.no_grad(): params = params - eta * params.grad
        losses.append(J.item())

    print(f'Epoch {epoch +1}, train loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in val_loader:
        x, y = batch
        #x : b x 1 x 28 x 28
        b = x.size(0)
        #x = x.view(b,-1)
        x = x.view(b,-1).cuda()
        with torch.no_grad():
            # Step 1 -> Forward
            l = model(x) # l: logits
        # Step 2 -> Compute the objective function
        #J = loss(l,y) # y = labels
        J = loss(l,y.cuda())
        losses.append(J.item())
    print(f'Epoch {epoch +1}, validation loss: {torch.tensor(losses).mean():.2f}')
