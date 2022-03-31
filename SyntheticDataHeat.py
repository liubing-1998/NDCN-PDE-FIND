import sys

import torch
import argparse
import datetime
import numpy as np
import networkx as nx
import torchdiffeq as ode
from utils_in_learn_dynamics import *


parser = argparse.ArgumentParser('Heat Diffusion Dynamic Case')

parser.add_argument('--seed', type=int, default=0, help='Random Seed')  # 随机种子
parser.add_argument('--n', type=int, default=400, help='Number of nodes')  # 节点个数
parser.add_argument('--network', type=str, choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')  # 网络类型
parser.add_argument('--viz', type=bool, default=True, help='save figure')  # 是否将数值保存成可视化图片
parser.add_argument('--T', type=float, default=5., help='Terminal Time')  # 时间的最终时刻
parser.add_argument('--time_tick', type=int, default=20)  # 生成多少时间点
parser.add_argument('--sparse', type=bool, default=False)  # 是否转换问稀疏矩阵进行计算
parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')

args = parser.parse_args()


# Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
n = args.n  # e.g nodes number 400  节点个数
N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
seed = args.seed  # 程序固定种子，可复现实验结果
# 生成邻接矩阵A
if args.network == 'grid':
    print("Choose graph: " + args.network)
    A = grid_8_neighbor_graph(N)
    G = nx.from_numpy_array(A.numpy())
elif args.network == 'random':
    print("Choose graph: " + args.network)
    G = nx.erdos_renyi_graph(n, 0.1, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'power_law':
    print("Choose graph: " + args.network)
    G = nx.barabasi_albert_graph(n, 5, seed=seed)
    G = networkx_reorder_nodes(G,  args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'small_world':
    print("Choose graph: " + args.network)
    G = nx.newman_watts_strogatz_graph(400, 5, 0.5, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'community':
    print("Choose graph: " + args.network)
    n1 = int(n/3)
    n2 = int(n/3)
    n3 = int(n/4)
    n4 = n - n1 - n2 - n3
    G = nx.random_partition_graph([n1, n2, n3, n4], .25, .01, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))

if args.viz:
    makedirs(r'figure/network/')
    # visualize_graph_matrix(G, args.network)

D = torch.diag(A.sum(1))
L = (D - A)

# 生成时间
t = torch.linspace(0., args.T, args.time_tick)  # 生成0-5的100个时间点
print(t)
# 是否转换为稀疏矩阵进行处理，
if args.sparse:
    # For small network, dense matrix is faster
    # For large network, sparse matrix cause less memory
    L = torch_sensor_to_torch_sparse_tensor(L)
    A = torch_sensor_to_torch_sparse_tensor(A)

# Initial Value 给一些节点一个初始值，相当于给一个初始化
x0 = torch.zeros(N, N)
x0[int(0.05*N):int(0.25*N), int(0.05*N):int(0.25*N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
x0[int(0.45*N):int(0.75*N), int(0.45*N):int(0.75*N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
x0[int(0.05*N):int(0.25*N), int(0.35*N):int(0.65*N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
x0 = x0.view(-1, 1).float()
energy = x0.sum()

# 热扩散方程
class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):  # 公式变换加x项也是可以学的
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f

with torch.no_grad():
    solution_numerical = ode.odeint(HeatDiffusion(L, 1), x0, t, method='dopri5')
    solution_numerical = torch.squeeze(solution_numerical)
    # print(solution_numerical.shape)

# 可视化
# 可视化保存的目录
if args.viz:
    dirname = r'figure/heat/' + args.network
    makedirs(dirname)
    fig_title = r'Heat Diffusion Dynamics'

now = datetime.datetime.now()
appendix = now.strftime("%m%d-%H%M%S")
zmin = solution_numerical.min()
zmax = solution_numerical.max()
for ii, xt in enumerate(solution_numerical, start=1):
    if args.viz and (ii % 10 == 0):
        pass
        # print(xt.shape)
        # visualize(N, x0, xt, '{:03d}-tru'.format(ii)+appendix, fig_title, dirname, zmin, zmax)


# # 需要的值
# # print(t)  # 时间点
# print(t.shape)  # 101
#
# # print(A)  # 邻接矩阵
# print(A.shape)  # 400*400
#
# # 邻接矩阵在计算过程中是用的L = (D - A)
# # print(L)
# print(L.shape)
#
# # print(solution_numerical)  # 求解的值，状态值
# print(solution_numerical.shape)  # 101*400

# 状态对时间导数值，用差分算或者直接套公式算
# 这里的t等价于Burgers.py中的t
# A等价于Burgers.py中的x
# solution_numerical等价于Burgers.py中的u

# 同一时刻所以有节点的状态，一次性计算，

from PDE_FIND import *

u = solution_numerical.numpy()  # 状态
t = t.numpy()  # 时间
x = L.numpy()  # 邻接矩阵
# print(u.shape, t.shape, x.shape)  # (101, 400) (101,) (400, 400)

X = np.arange(400)
X, T = np.meshgrid(X, t)
fig1 = plt.figure()
ax = fig1.gca(projection='3d')
surf = ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), linewidth=0, antialiased=False)
plt.title('Heat Network Equation', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', fontsize=16)
plt.show()


dt = t[1] - t[0]
# dx = x[2] - x[1]
# print(dt, dx)  # 0.05, 矩阵
# print(dx.shape)  400*1

# 调用网路上的基库构建函数
# build_networks_system(u, dt, D=3, time_diff='poly', lam_t=None, width_t=None, deg_t=None, sigma=2)
Ut, R, rhs_des = build_networks_system(u.T, dt, D=3, time_diff='FD', A=x)

print(Ut.shape, R.shape)
print(rhs_des)
# sys.exit()
w = TrainSTRidge(R, Ut, 10**-5, 1)
print("PDE derived using STRidge")
print(w)
print_pde(w, rhs_des)




