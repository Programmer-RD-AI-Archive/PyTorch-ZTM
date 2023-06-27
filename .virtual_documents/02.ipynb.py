import sklearn
from sklearn.datasets import make_circles
# Make 100 Samples
n_samples = 10000
X,y = make_circles(n_samples,noise=0.125,random_state=42)


import matplotlib.pyplot as plt
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


X[:5],y[:5]


# Make a DataFrame of circle data
import pandas as pd


circles = pd.DataFrame({"X1":X[:,0],"X2":X[:,1],"y":y})


# Visualizing
plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)


X_sample = X[0]
y_sample = y[0]
X_sample,y_sample,X_sample.shape,y_sample.shape


# Turn data into tensors
import torch
torch.__version__


X.dtype


X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)


X[:2],y[:2]


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


len(X_train),len(y_test)


import torch
from torch import nn

# Make device agnositic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


device


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2,1024) # 
        self.layer_2 = nn.Linear(1024,1)
    
    def forward(self,X):
        return self.layer_2(self.layer_1(X)) # x -> layer_1 -> layer_2


model_0 = CircleModelV0().to(device)


model_0


list(model_0.parameters())


model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=64),
    nn.Linear(64,1)
).to(device)


untrained_preds = model_0(X_test)


untrained_preds[0],y_test[0]



