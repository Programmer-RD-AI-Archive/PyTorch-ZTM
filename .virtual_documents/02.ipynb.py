import sklearn
from sklearn.datasets import make_circles
# Make 100 Samples
n_samples = 10000
X,y = make_circles(n_samples,noise=0.069,random_state=42)


import matplotlib.pyplot as plt
import numpy as np
import torch


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


X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)


X[:2],y[:2]


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)






