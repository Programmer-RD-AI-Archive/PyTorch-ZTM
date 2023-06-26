what_were_covering = {
    1:"Data (prepare and load)",
    2:"build model",
    3:"fitting the model to data (training)",
    4:"making predictions and evaluation (inference)",
    5:"saving and loading a model",
    6:"putting it all together"
}


import torch
from torch import nn # nn contains all of PyTorch's building blocs for neural networks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision

device = torch.device('cuda')

torch.__version__


# Create the known parameters

# y = bX + a
weight = 0.7 # b
bias = 0.3 # a

# Create data

start = 0.0
end = 1.0
step = 0.02
X = torch.arange(start, end ,step).unsqueeze(dim=1).to('cuda')
y = (weight * X + bias).to('cuda')


X[:5], y[:5]


len(X), len(y)


from sklearn.model_selection import train_test_split


train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split],y[:train_split]
X_test,y_test = X[train_split:], y[train_split:]


len(X_train),len(y_train), len(X_test), len(y_test)


def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    """
    Plots trainings data, test data and compares predictions
    """
    plt.figure(figsize=(10,7))
    
    # Plot trainings data in blue
    plt.scatter(train_data.cpu(),train_labels.cpu(), c='b', s=10, label='Trainings data')
    
    # Plot test data in green
    plt.scatter(test_data.cpu(), test_labels.cpu(), c='g', s=10, label='Testing data')
    
    if predictions is not None:
        # Plot the predictions
        plt.scatter(test_data.cpu(),predictions.cpu(),c='r', s=10, label='predictions')
    plt.legend(prop={"size":14})


plot_predictions()


# Create a linear regression model class

class LinearRegressionModel(nn.Module): # <- almost everythhing in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        
    def forward(self,X):
        return self.weights * X + self.bias


model = LinearRegressionModel().to('cuda')


torch.manual_seed(42)

# Create instance of the model
model_0 = LinearRegressionModel().to('cuda')

list(model_0.parameters()) # the values are the values they are becz we got random val


model_0.state_dict()


# predictions = inference
with torch.inference_mode():
    y_preds = model_0(X_test.to('cuda'))
plot_predictions(predictions=y_preds.cpu())


list(model_0.parameters()),model_0.state_dict()


loss_fn = nn.L1Loss()


optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01) 
# learning rate = possibily th emost important hyper parameter, it is a value that we set our selves, the learning rate is 
# smaller the learning rate the smaller the change done to parameters


from tqdm import tqdm
# the model gets to see the data once
epochs = 100
epoch_count = []
loss_values = []
test_loss_values = []


for epoch in tqdm(range(epochs)):
    # Set model to training model
    model_0.train() # turns on gradient tracking
    
    # Forward pass
    y_preds = model_0(X_train)
    
    # Calculaate the loss
    loss = loss_fn(y_preds,y_train)
    
    # Optimizer zero grad
    optimizer.zero_grad() # make sure that optimizer data doesnt add up, think as if its "C" so that it changes in every iteration
    
    # Back propagration
    loss.backward()
    
    # Gradient Descent
    optimizer.step()
    
    model_0.eval() # turns off  stf like dropout and batchnormalisation
    
    with torch.inference_mode(): # turns off gradient tracking 
        # 1. Do the forward pass
        test_pred = model_0(X_test)
        
        # Calculate the loss
        test_loss = loss_fn(test_pred,y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.cpu().detach())
        test_loss_values.append(test_loss.cpu().detach())
        print(f"Epoch: {epoch} | Test : {test_loss} | Loss : {loss}")


y_preds[:5],y_train[:5]


# predictions = inference
with torch.inference_mode():
    y_preds = model_0(X_test.to('cuda'))
plot_predictions(predictions=y_preds.cpu())


model_0.state_dict()


weight,bias


# Plot the loss curves
plt.plot(epoch_count,loss_values,label='Train Loss')
plt.plot(epoch_count,test_loss_values,label='Test Loss')
plt.title("Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()


# Saving out PyTorch model
torch.save(model.state_dict(),'./models/01/00.pth')


state_dict_00 = torch.load("./models/01/00.pth")


model_01 = LinearRegressionModel()


model_01.load_state_dict(state_dict_00)



