import sklearn
from sklearn.datasets import make_circles
# Make 100 Samples
n_samples = 10000
X,y = make_circles(n_samples,noise=0.0625,random_state=42)


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
        self.layer_1 = nn.Linear(2,2048) # 
        self.layer_2 = nn.Linear(2048,1)
        self.relu = nn.ReLU()
    
    def forward(self,X):
        # return self.layer_2(self.relu(self.layer_1(X))) # x -> layer_1 -> layer_2
        return self.layer_2(self.layer_1(X))


model_0 = CircleModelV0().to(device)


model_0


list(model_0.parameters())


# model_0 = nn.Sequential(
#     nn.Linear(in_features=2,out_features=64),
#     nn.Linear(64,1)
# ).to(device)


untrained_preds = model_0(X_test)


untrained_preds[0],y_test[0]


loss_fn = nn.BCEWithLogitsLoss() # has the sigmoid function builtin
# BCELoss() requries sigmoid to be builtin to the model itself


optimizer = torch.optim.Adam(model_0.parameters())


epochs = 10
batch_size = 32


# Calculate accuracy
def accuracy_fn(y_true,y_preds):
    correct = torch.eq(y_true,y_preds).sum().item() # gives a False True list -> Tensor no. of true > just normal no.
    acc = correct/len(y_preds)*100
    return acc


y_logits = model_0(X_test)


y_preds_probs = torch.sigmoid(y_logits)


y_preds_probs.round()


y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)))


y_preds = torch.round(y_preds_probs)


y_preds.squeeze()


test_loss_iter = []
train_loss_iter = []
train_accuracy_iter = []
test_accuracy_iter = []


from tqdm import tqdm


# get_ipython().run_line_magic("%time", "")
# epochs = 100
# batch_size = 32

# for epoch in tqdm(range(epochs)):
#     for i in range(0,len(X_train),batch_size):
#         X_batch = X_train[i:i+batch_size]
#         y_batch = y_train[i:i+batch_size]
#         preds = model_0(X_batch)
#         true_preds = torch.round(torch.sigmoid(preds.squeeze()))
#         loss = loss_fn(preds.squeeze(),y_batch.squeeze())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     with torch.inference_mode():
#         y_test_preds = model_0(X_test)
#         loss_test = loss_fn(y_test_preds.squeeze(),y_test.squeeze())
#         true_test_preds = torch.round(torch.sigmoid(y_test_preds))
#     train_loss_iter.append(loss.cpu().detach().numpy())
#     test_loss_iter.append(loss_test.cpu().detach().numpy())
#     train_accuracy_iter.append(accuracy_fn(y_batch,true_preds))
#     test_accuracy_iter.append(accuracy_fn(y_test,true_test_preds))


for epoch in tqdm(range(epochs)):
    model_0.train()
    y_logists = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logists,y_train)
    acc = accuracy_fn(y_true=y_train,y_preds=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,y_preds=test_pred)
        
print(f"""
        Loss : {loss}
        Accuracy : {acc}
        Test Loss : {test_loss}
        Test Accuracy : {test_acc}
        """)


import requests
from pathlib import Path

# Download helper functions from PyTorch repo
if not Path("helper_functions.py").is_file():
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)


from helper_functions import *


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0,X_test,y_test)


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(2,256)
        self.linear2 = nn.Linear(256,512)
        self.linear3 = nn.Linear(512,1024)
        self.linear4 = nn.Linear(1024,512)
        self.linear5_output = nn.Linear(512,1)
    
    def forward(self,X):
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))
        X = self.activation(self.linear3(X))
        X = self.activation(self.linear4(X))
        X = self.linear5_output(X)
        return X


model = ClassificationModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


epochs = 250
batch_size = 32


import wandb


wandb.init(project="02",name="Adjusted")
for epoch in tqdm(range(epochs)):
    for i in range(0,len(X_train),batch_size):
        torch.cuda.empty_cache()
        model.train()
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        preds = model(X_batch).squeeze()
        norm_preds = torch.round(torch.sigmoid(preds))
        loss = criterion(preds,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_preds = model(X_train).squeeze()
            test_preds = model(X_test).squeeze()
            loss_test = criterion(test_preds,y_test)
            loss_train = criterion(train_preds,y_train)
            train_preds = torch.round(torch.sigmoid(train_preds))
            test_preds = torch.round(torch.sigmoid(test_preds))
            acc_train = accuracy_fn(y_train,train_preds)
            acc_test = accuracy_fn(y_test,test_preds)
            wandb.log({
                "Train Loss":loss_train,
                "Test Loss":loss_test,
                "Train Accuracy": acc_train,
                "Test Accuracy": acc_test
            })
wandb.finish()


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model,X_test,y_test)


torch.max(torch.tensor(0),torch.tensor(4))


# Create a tensor
A = torch.arange(-10,10)


A.dtype


A


plt.plot(A)


plt.plot(torch.relu(A))


def relu(X):
    return torch.max(torch.tensor(0),X)


plt.plot(relu(A))


def sigmoid(X):
    return 1 / (1  + torch.exp(-X))


plt.plot(sigmoid(A))


torch.exp(torch.tensor(1))


from sklearn.datasets import make_blobs


NUM_CLASSES = 4  
NUM_FEATURES = 2


X,y = make_blobs(n_samples=1000,n_features=NUM_FEATURES,centers=NUM_CLASSES,cluster_std=1,random_state=42)
plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.RdYlBu)
# X is the cordinate spaces and y is used to assign each of the points a class (label)


X,y = torch.from_numpy(X).type(torch.float).to(device),torch.tensor(y).type(torch.float).to(device)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.125)


class BlobModel(nn.Module):
    def __init__(self,input_features=2,output_features=4,hidden_units=1024):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units * 2),
            nn.ReLU(),
            nn.Linear(hidden_units * 2, output_features)
        )
    
    def forward(self,X):
        return self.linear_layer_stack(X)


model = BlobModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


epochs = 100
batch_size = 32


y_preds = model(X_test)


y_pred_probs = torch.softmax(y_preds,dim=1)


torch.argmax(y_pred_probs[0]),y_pred_probs[0]


# Conver models preditions probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs,dim=1)


y_pred_probs.shape


y_preds


# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs 
epochs = 100

# Put data to the target device
X_blob_train, y_blob_train = X_train.to(device), y_train.to(device)
X_blob_test, y_blob_test = X_test.to(device), y_test.to(device)

# Loop through data
for epoch in range(epochs):
  ### Training 
  model.train()

  y_logits = model(X_blob_train)
  y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

  loss = loss_fn(y_logits, y_blob_train)
  acc = accuracy_fn(y_true=y_blob_train,
                    y_pred=y_pred)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  ### Testing
  model.eval()
  with torch.inference_mode():
    test_logits = model(X_blob_test)
    test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
    
    test_loss = loss_fn(test_logits, y_blob_test)
    test_acc = accuracy_fn(y_true=y_blob_test,
                           y_pred=test_preds)
    
  # Print out what's happenin'
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}get_ipython().run_line_magic("")", "")



