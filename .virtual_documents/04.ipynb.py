import torch
from torch import nn,optim
torch.__version__


device = 'cuda' if torch.cuda.is_available() else 'cpu'


import requests
import zipfile
from pathlib import Path

# Setup path to a data folder

data_path = Path("data/04/01")

data_path.mkdir(parents=True,exist_ok=True)

with open(data_path / "data.zip",'wb') as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    f.write(request.content)
with zipfile.ZipFile(data_path / "data.zip", 'r') as zip_ref:
    zip_ref.extractall(data_path)


import os
def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} and {len(filenames)} images in {dirpath}")


walk_through_dir(data_path)





import random
from PIL import Image
image_path_list = list(data_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
# str(random_image_path).split("/")[-2]
image_class =  random_image_path.parent.stem
img = Image.open(random_image_path)
# 5. Print metadata
img


random_image_path,image_class,img.height,img.width,



import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(10,7))
plt.imshow(np.asarray(img))
plt.axis(False);


import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms


data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


plt.imshow(torch.permute(data_transform(img),(1,2,0)))


def plot_transformed_images(image_paths,transform,n=3,seed=42):
    """
    Selects random iamges  from a pth of images and loads/transforms
    them then plots the original vs the transformed version
    """
    random_image_paths = random.sample(image_paths,k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].axis(False)
            ax[1].imshow(torch.permute(transform(f),(1,2,0)))
            ax[1].axis(False)
            fig.suptitle(f"Class : {image_path.parent.stem}",fontsize=16)


plot_transformed_images(image_path_list,data_transform)


train_dir = "data/04/01/train/"
test_dir = "data/04/01/test/"


# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir,transform=data_transform)
test_data = datasets.ImageFolder(root=test_dir,transform=data_transform)


train_data,test_data


# Get class names
class_names = train_data.classes
class_names


class_dict = train_data.class_to_idx
class_dict


img,label = train_data[0][0],train_data[0][1]


img.shape,img.dtype


label,type(label)


img_permute = img.permute(1,2,0)


plt.imshow(img_permute)


import os
os.cpu_count()


train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=os.cpu_count())
test_dataloader = DataLoader(test_data,batch_size=32,shuffle=True,num_workers=os.cpu_count())


train_dataloader,test_dataloader



img,label = next(iter(train_dataloader))


img.shape


plt.imshow(img[0].permute(1,2,0))


import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple,Dict,List


# Setup path for target directory
target_directory = train_dir


class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
class_names_found


def find_classes(directory:str) -> Tuple[List[str], Dict[str,int]]:
    # Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldnt finda ny classes  in {directory} ... please check file strucutre")
    class_to_idx = {class_name:i for i, class_name in enumerate(classes)}
    return classes,class_to_idx


find_classes(target_directory)


os.listdir("data/04/01/test")


# Write a custom dataset class
from torch.utils.data import Dataset
# Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    def __init__(self,targ_dir:str,transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes,self.class_to_idx = find_classes(targ_dir)
    
    def load_image(self,index) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index) -> Tuple[torch.tensor,int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            img = self.transform(img)
        return (img,class_idx)


train_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])



train_data_custom = ImageFolderCustom(target_directory,train_transforms)
test_data_custom = ImageFolderCustom(target_directory,test_transforms)


train_data_custom,test_data_custom


from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom, batch_size=32,num_workers=os.cpu_count())
test_dataloader_custom = DataLoader(dataset=test_data_custom, batch_size=32,num_workers=os.cpu_count())


# Let's look at trivialaugment
from torchvision import transforms
# train_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     # transforms.TrivialAugmentWide(num_magnitue_bins=31),
#     transforms.ToTensor()
# ])


import torchvision
torchvision.__version__


simple_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])
train_ds = ImageFolderCustom("data/04/01/train/",transform =simple_transform)
test_ds = ImageFolderCustom("data/04/01/test/",transform =simple_transform)
train_dl = DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=os.cpu_count())
test_dl = DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=os.cpu_count())


class TinyVGG(nn.Module):
    def __init__(self,input_chanels,output_clzes):
        super().__init__()
        self.convblo1 = nn.Sequential(
            nn.Conv2d(input_chanels,10,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.convblo2 = nn.Sequential(
            nn.Conv2d(10,10,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convblo3 = nn.Sequential(
            nn.Conv2d(10,10,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.convblo4 = nn.Sequential(
            nn.Conv2d(10,10,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*16*16,output_clzes)
        )
        
    def forward(self,X):
        # print(X.shape)
        y = self.convblo1(X)
        y = self.convblo2(y)
        y = self.convblo3(y)
        y = self.convblo4(y)
        # print(y.shape)
        y = self.output(y)
        return y


next(iter(test_dl))[0].shape,next(iter(train_dl))[0].shape


model = TinyVGG(3,len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs = 100


def accuracy(model,loader):
    tot = 0
    no = 0
    with torch.inference_mode():
        for X,y in loader:
            preds = model(X.to(device))
            preds = torch.argmax(torch.softmax(preds,dim=1),dim=1)
            tot_iter = 0
            cor = 0
            for pred,y_iter in zip(preds,y):
                if pred == y_iter:
                    cor += 1
                tot_iter += 1
            tot += cor/tot_iter
            no += 1
    return (tot/no)*100


def loss_fn(model,loader,criterion):
    tot = 0
    no = 0
    with torch.inference_mode():
        for X,y in loader:
            preds = model(X.to(device))
            loss = criterion(preds.to(device),y.long().to(device))
            tot += loss.item()
            no += 1
    return tot/no


import wandb
from tqdm import tqdm


torch.manual_seed(42)
torch.cuda.manual_seed(42)
wandb.init(project="04",name="Model:0")
for epoch in tqdm(range(epochs)):
    for X_batch,y_batch in train_dl:
        X_batch = torch.tensor(X_batch).to(device).float()
        y_batch = torch.tensor(y_batch).to(device).long()
        preds = model(X_batch)
        loss = criterion(preds,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    wandb.log({
        "Train Loss":loss_fn(model,train_dl,criterion),
        "Test Loss":loss_fn(model,test_dl,criterion),
        "Train Accuracy":accuracy(model,train_dl),
        "Test Accuracy":accuracy(model,test_dl)
    })
wandb.finish()


get_ipython().getoutput("pip install torchinfo")


import torchinfo
from torchinfo import summary


summary(model,input_size=(32,3,64,64))


def train_step(model:torch.nn.Module,
               datalaoder:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer
              ):
    model.train()
    train_loss, train_acc = 0,0
    for batch,(X,y) in enumerate(datalaoder):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    train_loss = train_loss/len(datalaoder)
    train_acc = train_acc/len(datalaoder)
    return train_loss,train_acc


def test_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module):
    tot_loss, tot_acc = 0,0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)
            y_pred = model(X)
            tot_loss += criterion(y_pred,y).item()
            tot_acc += (torch.softmax(y_pred,dim=1).argmax(dim=1)==y).sum().item()/len(y_pred)
    return tot_loss / len(dataloader), tot_acc / len(dataloader)


def train(model,epochs,test_datalaoder,train_dataloader,loss_fn,optimizer):
    for epoch in tqdm(range(epochs)):
        print(epoch)
        results = {
            "train_loss":[],
            "train_acc":[],
            "test_loss":[],
            "test_acc":[]
        }
        train_loss,train_acc = train_step(model,train_dataloader,loss_fn,optimizer)
        test_loss,test_acc = test_step(model,test_dataloader,loss_fn)
        print(f"""
                Train Loss : {train_loss}
                Test Loss : {test_loss}
                Train Accuracy : {train_acc}
                Test Accuracy : {test_acc}
        """)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
    return results


torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = TinyVGG(3,len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
results = train(model,5,test_dl,train_dl,criterion,optimizer)


# Get the model_0_result keys

results.keys()


def plot_loss_curves(results:Dict[str,List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    acc = results['train_acc']
    test_acc = results['test_acc']
    no_epochs = range(5)
    print(no_epochs,test_acc)
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label='train_loss')
    plt.plot(epochs,test_loss,label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs,acc,label='train_acc')
    plt.plot(epochs,test_acc,label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


plot_loss_curves(results)


# get_ipython().getoutput("pip install torchvision --upgrade")


train_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomGrayscale(p=0.125),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(random.randint(1,50)),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])


train_ds = ImageFolderCustom("data/04/01/train/",transform =train_transform)
test_ds = ImageFolderCustom("data/04/01/test/",transform =test_transform)
train_dl = DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=os.cpu_count())
test_dl = DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=os.cpu_count())


torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = TinyVGG(3,len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
results = train(model,5,test_dl,train_dl,criterion,optimizer)


plot_loss_curves(results)


# custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))



