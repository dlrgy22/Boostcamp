import torch
import numpy as np

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype = torch.long)
print(x)

x = torch.tensor([5.5, 3.3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
y = torch.randn_like(x, dtype=torch.float)
print(x)
print(y)
print(x + y)

print(torch.add(x, y))
result = torch.empty_like(x)
torch.add(x, y, out=result)
print(result)

# in place 방식 뒤에는 _가 붙는다.
y.add_(x)
print(y)

x = torch.randn(4, 4)
y = x.view(16, 1)
z = x.view(-1, 2)
print(x.size(), y.size(), z.size())

x = torch.tensor([1])
print(x.item())

a = torch.ones(5)
print(a)

b = a.numpy()
print(type(b))
a.add_(1)
print(a)
print(b)

numpy_array = np.array([1, 2, 3, 4])
torch_tensor = torch.from_numpy(numpy_array)
print(numpy_array)
print(torch_tensor)
np.add(numpy_array, 1, out = numpy_array)
print(numpy_array)
print(torch_tensor)