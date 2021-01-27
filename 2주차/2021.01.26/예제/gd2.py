import numpy as np

#데이터
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# 정답레이블 *1 + *2 + 3
y = np.dot(X, np.array([1, 2])) + 3
Y = np.array([[1, 2], [2, 3], [4, 5]])
# 초기 gradient 벡터
beta_gd = [10.1, 15.1, -6.5]
#상수항 추가
X_ = np.array([np.append(x, [2]) for x in X])
for t in range(5000):
    error = y - X_ @ beta_gd
    grad = -np.transpose(X_) @ error
    beta_gd = beta_gd - 0.01 * grad

print(beta_gd)
