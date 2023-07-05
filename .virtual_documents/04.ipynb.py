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
