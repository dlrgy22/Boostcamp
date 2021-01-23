import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z, out)

out.backward()
print(x.grad)

print("\n########################################################################################################################\n")

x = torch.randn(3, requires_grad=True)
print(x)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
print(y)
y.backward(v)

print(x.grad)