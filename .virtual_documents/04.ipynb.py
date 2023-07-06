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
    transforms.Resize(size=(128,128)),
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



