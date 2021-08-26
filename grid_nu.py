import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import matplotlib as mpl

mpl.style.use('classic')
font = {'family' : 'serif', 'weight' : 'normal', 'size' : 20}
plt.rc('font', **font)

Nx = 64
Ny = Nx
Nz = Nx

N = np.arange(0, Nx+2, 1)

hx, hy, hz = 1/(Nx), 1/(Ny), 1/(Nz)

x = np.linspace(0, 1 + hx, Nx + 2, endpoint=True) - hx/2
y = np.linspace(0, 1 + hx, Ny + 2, endpoint=True) - hy/2
z = np.linspace(0, 1 + hx, Nz + 2, endpoint=True) - hz/2

beta = 1.0e0

xb = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*x))/np.tanh(beta))
yb = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*y))/np.tanh(beta))
zb = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*z))/np.tanh(beta))	

dx = np.diff(xb)
print (min(dx), max(dx), (max(dx)/min(dx)))

plt.figure(1)
plt.scatter(N[1:Nx+1], xb[1:Nx+1])
plt.axhline(y=0.1, lw=1.5, color="k", ls='-.')

plt.xlabel("$N$", fontsize = 30)
plt.ylabel("$x$", fontsize = 30)

plt.xlim(1,Nx)
plt.ylim(0,1)
plt.tight_layout()

plt.show()






			
			

			
			 

