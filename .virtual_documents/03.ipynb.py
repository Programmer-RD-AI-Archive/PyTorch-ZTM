import torch
import torchvision
from torch import nn, optim
from torchvision import datasets,models,transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


torch.__version__,torchvision.__version__


# Setup training data
train_data = datasets.FashionMNIST(root="data/03/",train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="data/03/",train=False,download=True,transform=transforms.ToTensor())



len(train_data),len(test_data)


# See the first training example
image,label = train_data[0]


image.shape,label


class_names = train_data.classes


class_names


class_to_idx = train_data.class_to_idx


class_to_idx


image.shape,label # C,H,W


plt.imshow(image.view(28,28,1),cmap="gray")
plt.title(class_names[label])
plt.axis(False);


# Plot more images
# torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
row,cols = 4,4
for i in range(1,row*cols+1):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(row,cols,i)
    plt.imshow(img.view(28,28,1),cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)


BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)


train_features_batch,train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape,train_labels_batch.shape


rdm_idx = torch.randint(0,len(train_features_batch),size = [1]).item()
img,label = train_features_batch[rdm_idx],train_labels_batch[rdm_idx]
plt.imshow(img.squeeze(),cmap="gray")
plt.title(class_names[label])
plt.axis(False)


# Create a flatten layer
flatten_model = nn.Flatten()
x = train_features_batch[0]
x.shape,flatten_model(x).shape


from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self,input_shape:int,hidden_units: int, output_shape:int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # No learning parameters
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    
    def forward(self,X) -> torch.Tensor():
        return self.layer_stack(X)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(42)

# Setup model with input parameters

model_0 = FashionMNISTModelV0(input_shape=28*28,hidden_units=2048,output_shape=len(class_names))


optimizer = optim.Adam(model_0.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


dummy_X = torch.rand([1,1,28,28])
model_0(dummy_X)



