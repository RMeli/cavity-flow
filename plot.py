import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib import cm

with open(sys.argv[1]) as f:
    Lx = float(f.readline().split()[-1])
    Ly = float(f.readline().split()[-1])
    Nx = int(f.readline().split()[-1])
    Ny = int(f.readline().split()[-1])

p = np.loadtxt(sys.argv[1])
u = np.loadtxt(sys.argv[2])
v = np.loadtxt(sys.argv[3])
name = sys.argv[4]

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Lx, Ny)

X, Y = np.meshgrid(x, y, indexing="xy")

plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)
plt.streamplot(X, Y, u, v)

plt.savefig(f"cavity-{name}.png")
