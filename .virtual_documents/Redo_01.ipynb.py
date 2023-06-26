import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.__version__


# y = mx+c
m = 0.69
c = 0.96
X = torch.arange(0,1,0.002).to(device)
y = torch.tensor(m*X+c).to(device)


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
    
    def forward(self,X):
        return X*self.m+self.c


model_0 = LinearRegressionModel().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model_0.parameters(),lr=0.01)


epochs = 100


m_iter = []
c_iter = []
train_loss_iter = []
test_loss_iter = []


for epoch in range(epochs):
    # train
    model_0.train()    
    y_preds = model_0(X_train)
    loss = criterion(y_preds,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # test
    model_0.eval()
    with torch.inference_mode():
        y_test_preds = model_0(X_test)
        test_loss = criterion(y_test_preds,y_test)
    paramters = dict(model_0.state_dict())
    m_iter.append(paramters['m'].cpu())
    c_iter.append(paramters['c'].cpu())
    train_loss_iter.append(loss.cpu().detach().numpy())
    test_loss_iter.append(test_loss.cpu().detach().numpy())


model_0.state_dict()


plt.plot(range(epochs),m_iter)
plt.xlabel('epochs')
plt.ylabel('m_iter')
plt.title("M_iter over time")
plt.show()


plt.plot(range(epochs),c_iter)
plt.xlabel('epochs')
plt.ylabel('c_iter')
plt.title("c_iter over time")
plt.show()


plt.plot(range(epochs),test_loss_iter,'b')
plt.plot(range(epochs),train_loss_iter,'r')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title("Losss over time")
plt.legend()
plt.show()


y_preds[5:7],y_train[5:7]


state_dict = model_0.state_dict()


torch.save(state_dict,'./models/01/redo_model_00.pth')


model_01 = LinearRegressionModel()


model_01.load_state_dict(torch.load('./models/01/redo_model_00.pth'))



