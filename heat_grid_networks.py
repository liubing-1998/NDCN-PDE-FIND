# demonstrates PDE-FIND on Burger's equation with an added diffusive term
# u_t + uu_x = u_{xx}
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PDE_FIND import *
import scipy.io as sio
import itertools
import matplotlib.pyplot as plt

####加载数据####
data = sio.loadmat('./Datasets/burgers.mat')
# u = real(data['usol'])  # 源代码应该是错了，没有加np.
# x = real(data['x'][0])
# t = real(data['t'][:,0])
u = np.real(data['usol'])
x = np.real(data['x'][0])
t = np.real(data['t'][:, 0])
# print(u.shape)  # (256, 101)
# print(x.shape)  # 256
# print(t.shape)  # 101
# print(x)  # -8 ~ 8 均等分
# print(t)  # 0 ~ 10 均等分
dt = t[1] - t[0]
dx = x[2] - x[1]
# print(dt, dx)  # 0.1 0.0625

####显示数据####
X, T = np.meshgrid(x, t)
fig1 = plt.figure()
ax = fig1.gca(projection='3d')
surf = ax.plot_surface(X, T, u.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), linewidth=0, antialiased=False)
plt.title('Burgers Equation', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=16)
# plt.show()
# print(np.max(u), np.min(u))  # 1.0 -3.4963163891843507e-09


####construct O(U) and compute U_t####，构建基库值并计算U对t的导数
####调用build_linear_system
Ut, R, rhs_des = build_linear_system(u, dt, dx, D=3, P=3, time_diff='FD', space_diff='FD')
print(['1'] + rhs_des[1:])
print(Ut.shape, R.shape)  # (25856, 1) (25856, 16)

# Solve with STRidge
w = TrainSTRidge(R, Ut, 10**-5, 1)
print("PDE derived using STRidge")
print_pde(w, rhs_des)

####计算误差####
err = abs(np.array([(1 - 1.000987)*100, (.1 - 0.100220)*100/0.1]))
print("Error using PDE-FIND to identify Burger's equation:\n")
print("Mean parameter error:", np.mean(err), '%')
print("Standard deviation of parameter error:", np.std(err), '%')

####增加噪声，并且采用插值方法求偏导数值，
np.random.seed(0)
un = u + 0.01*np.std(u)*np.random.randn(u.shape[0], u.shape[1])

Utn, Rn, rhs_des = build_linear_system(un, dt, dx, D=3, P=3, time_diff='poly', deg_x=4, deg_t=4, width_x=10, width_t=10)

# Solve with STRidge
w = TrainSTRidge(Rn, Utn, 10**-5, 1)
print("PDE derived using STRidge")
print_pde(w, rhs_des)

err = abs(np.array([(1 - 1.009655)*100, (.1 - 0.102966)*100/0.1]))
print("Error using PDE-FIND to identify Burger's equation with added noise:\n")
print("Mean parameter error:", np.mean(err), '%')
print("Standard deviation of parameter error:", np.std(err), '%')

