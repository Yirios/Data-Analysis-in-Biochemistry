import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op

# 需要拟合的数据组
x_group = np.array([3, 6.1, 9.1, 11.9, 14.9])
y_group = np.array([0.0221, 0.0491, 0.0711, 0.0971, 0.1238])


# 需要拟合的函数
def f_1(x, A, B):
    return A * x + B


# 得到返回的A，B值
A, B = op.curve_fit(f_1, x_group, y_group)[0]
print(A,B)
# 数据点与原先的进行画图比较
plt.scatter(x_group, y_group, marker='o',label='real')
x = np.arange(0, 15, 0.01)
y = A * x + B
plt.plot(x, y,color='red',label='curve_fit')
plt.legend()
plt.title('%.5fx%.5f=y' % (A, B))
plt.show()