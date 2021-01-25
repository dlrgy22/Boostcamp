import numpy as np

a = [1, 2, 3, 4, 5]
b = [5, 4, 3, 2, 1]

np_a = np.array(a, int)
np_b = np.array(b, int)
print(a)

test_array = np.array(["1", "4", 5.0, 8], float)
print(type(test_array))

print(a[0] is b[-1])
print(np_a[0] is np_b[-1])

print(type(np_a))
print(np_a.shape)
print(np_a.dtype)

test_matrix = np.arange(3)
result = np.ones_like(test_matrix)
print(result)

three_identity = np.identity(n = 3, dtype=np.int8)
print(three_identity)

#예시
print(np.eye(N=3, M=5, k=1, dtype=np.int8))

test_matrix = np.arange(9).reshape(3, 3)
print(np.diag(test_matrix, k = 1))

#예제
test_a = np.array([1, 3, 0], float)
test_b = np.array([5, 2, 1], float)
print(test_a > test_b)

a = np.array([1, 3, 0], float)
print(np.logical_and(a > 0, a < 3))

a = np.array([1, 3, 0], float)
print(np.where(a > 0, 3, 2))

a = np.array([1, 3, 0], float)
print(np.where(a > 0))

a = np.array([[1, 2, 4, 7], [9, 88, 6, 45], [9, 76, 3, 4]])
print(np.argmax(a, axis=1), np.argmin(a, axis=0))

X = np.array([[1, -2, 3],
              [7, 5, 0],
              [-2, -1, 2]])
print(np.linalg.inv(X))
print(X @ np.linalg.inv(X))
