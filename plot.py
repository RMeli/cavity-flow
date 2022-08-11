import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib import cm

p = np.loadtxt(sys.argv[1])
u = np.loadtxt(sys.argv[2])
v = np.loadtxt(sys.argv[3])

x = np.linspace(0, 2, 41)
y = np.linspace(0, 2, 41)

X, Y = np.meshgrid(x, y, indexing="xy")

plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)
plt.streamplot(X, Y, u, v)

plt.savefig("cavity.png")