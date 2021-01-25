import numpy as np

def l1_norm(x):
	x_norm = np.abs(x)
	x_norm = np.sum(x_norm)
	return x_norm

def l2_norm(x):
	x_norm = x*x
	x_norm = np.sum(x_norm)
	x_norm = np.sqrt(x_norm)
	return x_norm

def angle(x, y):
	v = np.inner(x, y) / (l2_norm(x) * l2_norm(y))
	theta = np.arccos(v)
	return theta

x = np.array([0, 1])
y = np.array([0, 2])
print(angle(x, y))

print(np.inner([1, -1, 1, -1], [4, -4, 4, -4]))

x = np.array([[1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]])
print(np.linalg.inv(x))

x = np.array([[1, 0, 1],
            [0, 1, 0]])
print(np.linalg.pinv(x))