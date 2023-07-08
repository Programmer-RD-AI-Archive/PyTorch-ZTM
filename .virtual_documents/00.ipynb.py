import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)


torch.device('cuda')


get_ipython().getoutput("nvidia-smi")


# scalar

scalar = torch.tensor(7)
scalar


scalar.ndim # finding the no. of dimentions


scalar.item() # just get the data in the tensor


# Vector
vector = torch.tensor([7,7])
vector


vector.ndim


vector.shape


# Matrix
MATRIX = torch.tensor(
    [
        [7,8],
        [8,9]
    ]
)


MATRIX.shape


MATRIX.ndim


# Tensor

TENSOR = torch.tensor([[[1,2,3,4],[5,6,5,4],[6,8,9,4]]])


TENSOR


TENSOR.shape # first dimension, second is row, third is element in row


TENSOR.ndim


TENSOR[0][2]


rdm_tensor = torch.rand(3,4) # rows, columns


rdm_tensor.shape


rdm_tensor.ndim


rdm_tensor


# Create random tensor with similar shape to image tensor
random_image_size_tensor = torch.rand(size=(3,484,224)) # height, width, color
random_image_size_tensor.shape,random_image_size_tensor.ndim


len(random_image_size_tensor[0]),len(random_image_size_tensor[1])


ones_tensor = torch.ones(3,4)
ones_tensor


zeros_tensor = torch.zeros(3,4)
zeros_tensor


rdm_tensor * zeros_tensor


ones_tensor.dtype


one_to_ten = torch.arange(start=1,end=11,step=1)


one_to_ten



ten_zeros = torch.zeros_like(input=one_to_ten)


ten_zeros


# Float 32 tensor
float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype=None,device='cuda',requires_grad=True)
float_32_tensor
# lower the bit no. the faster the data can be proccessed
# requires_grad means whther or not ot track the tensors gradient


float_32_tensor.dtype


float_16_tensor = float_32_tensor.type(torch.float16)


float_16_tensor


float_16_tensor * float_32_tensor


# Create a tensor
some_tensor = torch.rand(3,4)
some_tensor


print(some_tensor)
print(some_tensor.dtype)
print(some_tensor.size())
print(some_tensor.device)


# Addition

tensor = torch.tensor([1,2,3])
tensor + 10


# Multiplication

tensor * 10


# Subtraction

tensor - 10


# Division

tensor / 1.5


# Torch builtin functions

torch.mul(tensor,10)
torch.div(tensor,1.5)
torch.add(tensor,10)
torch.subtract(tensor,10)


get_ipython().run_cell_magic("time", "", """torch.matmul(tensor, tensor)""")


tensor*tensor


tensor


get_ipython().run_cell_magic("time", "", """# Matricx multiplication by hadnd
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]""")


tensor_A = torch.tensor([
        [1,2],
        [3,4],
        [5,6]
])

tensor_B = torch.tensor([
        [7,8],
        [8,11],
        [9,12]
])


torch.matmul(tensor_A,tensor_B.view(2,3))


tensor_A.T, tensor_A


tensor_B.T,tensor_B


x = torch.arange(0,100,10)


x


x.min()


x.max()


# requires a tensor of float or complex types
x.type(torch.float32).mean()


x.sum()


x.argmax()


x.argmin()


x = torch.arange(1,10)
x


x.shape


x_reshape = x.reshape(1,9)


x_reshape


z = x.view(1,9)


z.shape


x.shape


# view shares the same memory
z[0][0] = 5
z,x


# Stack tensors on top
x_stacked = torch.stack([x,x,x,x],dim=1)


x_stacked


# torch.squeeze() remove all 1 dimensional shape


x_reshape


x_reshape.shape


x_reshape.squeeze().shape


x_squeezed = x_reshape.squeeze()


x_squeezed.unsqueeze(dim=0)


# premute - changes the dimensions of a tensors


x_original = torch.rand(size=(224,224,3)) #height, width ,color channels


torch.permute(x_original,(2,0,1)).shape, x_original.shape


# Create a tensor

x = torch.arange(1,10).reshape(1,3,3)


x


x[0]


x[0][0]


x[0][0][0]


x[0][2][2]


x[:,:,2]


# NumPy array to tensor
import numpy as np

array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) # when converting from numpy, pyotrch reflects numpy's default datatype of float64
array,tensor


tensor.dtype


array = array +1


tensor = torch.ones(7)
numpy_tensor = tensor.numpy()


numpy_tensor,tensor


random_A = torch.rand(3,4)
random_B = torch.rand(3,4)

print(random_A)
print(random_B)
print(random_A == random_B)


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_A = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
random_B = torch.rand(3,4)

print(random_A)
print(random_B)
print(random_A == random_B)


torch.cuda.is_available()



