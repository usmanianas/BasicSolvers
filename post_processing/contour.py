	
import numpy as np
from numpy import *
import pylab as plt
import h5py
from matplotlib import cm

import Vector_Plot as vp
import derivative as der
import modal_energy_2D as mode
	
	
#filename = "../no_slip/Ra1e5/Ta1e6/Soln_0475.0000.h5"
filename = "../no_slip/Ra1e5/Ta1e6/decay/nu3em3/Ro1/Soln_0496.0000.h5"
#filename = "../const_stress/stress_3/Soln_0500.0000.h5"
#filename = "../free_slip_yz/Ra1e5/Ta1e6/Soln_1000.0000.h5"

Ra, Pr, Ta = (1.0e5, 1.0, 1.0e6)

def hdf5_reader(filename,dataset):
	file_V1_read = h5py.File(filename)
	dataset_V1_read = file_V1_read["/"+dataset]
	V1=dataset_V1_read[:,:,:]
	return V1
	
Vx = hdf5_reader(filename, "Vx")
Vy = hdf5_reader(filename, "Vy")
Vz = hdf5_reader(filename, "Vz")
P = hdf5_reader(filename, "P")
#T = hdf5_reader(filename, "T")

Nx, Ny, Nz = (Vx.shape[0], Vx.shape[1], Vx.shape[2])

print ("Nx=",Nx, "Ny=",Ny, "Nz=",Nz)


#divU = (der.delX(Vx)[0]+der.delX(Vy)[1]+der.delX(Vz)[2])
###print unravel_index(divU.argmax(), divU.shape), np.amax(divU)
###print mean(abs(divU))
#
#hf = h5py.File('delU.h5', 'w')
#hf.create_dataset('divergence', data = divU)
#hf.close()


x = np.linspace(0, 1, Nx, endpoint=True)		
y = np.linspace(0, 1, Ny, endpoint=True)
z = np.linspace(0, 1, Nz, endpoint=True)	

dx = 1.0/(Nx-1.0)	
dy = 1.0/(Ny-1.0)		
dz = 1.0/(Nz-1.0)		

def dF_dx(F):
	dX_dx = np.zeros([Nx, Ny, Nz])
	for i in range(1,Nx-1):
		for j in range(1,Ny-1):
			for k in range(1,Nz-1):
				dX_dx[i,j,k] = (F[i+1,j,k]-F[i-1,j,k])/(2.0*dx)								
	return dX_dx

def dF_dy(F):
	dX_dy = np.zeros([Nx, Ny, Nz])
	for i in range(1,Nx-1):
		for j in range(1,Ny-1):
			for k in range(1,Nz-1):
				dX_dy[i,j,k] = (F[i,j+1,k]-F[i,j-1,k])/(2.0*dy)								
	return dX_dy
	
def dF_dz(F):
	dX_dz = np.zeros([Nx, Ny, Nz])
	for i in range(1,Nx-1):
		for j in range(1,Ny-1):
			for k in range(1,Nz-1):
				dX_dz[i,j,k] = (F[i,j,k+1]-F[i,j,k-1])/(2.0*dz)								
	return dX_dz		

omega_x = dF_dy(Vz) - dF_dz(Vy)   #dVz_dy - dVy_dz	
#omega_y = dF_dz(Vx) - dF_dx(Vz)   #dVx_dz - dVz_dx
#omega_z = dF_dx(Vy) - dF_dy(Vx)   #dVy_dx - dVx_dy


# Contour plots ###############
pl = int(Nx/2)    #plane location in terms of grid number
vs = 8         # vector size

print ("Contour plot of Vx at x =", x[pl])
Z, Y = np.meshgrid(y,z)
plt.figure(1)
#clev = np.arange(Vy[xp, :, :].min(), Vy[xp, :, :].max(), 0.0002)
#print (T[xp, :, :].max())

#cp = plt.pcolor(Y, Z, omega_x[pl, :, :], cmap=cm.coolwarm)      # yz plane

cp = plt.contourf(Y, Z, omega_x[pl, :, :], 500, cmap=cm.coolwarm)      # yz plane
#cp = plt.contourf(Y, Z, omega_z[:, pl, :], 500, cmap=cm.seismic)	    # xz plane
#cp = plt.contourf(Y, Z, omega_z[:, :, pl], 500, cmap=cm.seismic)		# xy plane
clb = plt.colorbar(ticks=[-0.3,-0.2,-0.1,0,0.1,0.2,0.3])

#clb = plt.colorbar()

quiv = plt.quiver(Y[0:Nx:vs, 0:Ny:vs], Z[0:Nx:vs, 0:Ny:vs], Vy[pl, 0:Nx:vs, 0:Ny:vs,], Vz[pl, 0:Nx:vs, 0:Ny:vs])            # yz plane
#quiv = plt.quiver(Y[0:Nx:vs, 0:Ny:vs], Z[0:Nx:vs, 0:Ny:vs], Vx[0:Nx:vs, pl, 0:Ny:vs,], Vz[0:Nx:vs, pl, 0:Ny:vs])            # xz plane
#quiv = plt.quiver(Y[0:Nx:vs, 0:Ny:vs], Z[0:Nx:vs, 0:Ny:vs], Vx[0:Nx:vs, 0:Nx:vs, pl], Vy[0:Ny:vs, 0:Nx:vs, pl])               # xy plane

#plt.imshow(T[ xp, :, :], cmap=cm.coolwarm, origin='lower', extent=[Y.min(), Y.max(), Z.min(), Z.max()])
#vmin = 0., vmax = 3.
plt.axis('scaled')
#clb.set_label(r'$\omega_x$', labelpad=-40, y=1.05, rotation=0)	
clb.ax.set_title(r'$\omega_x$', fontsize = 25)	
#plt.xlim(0.0,1)
#plt.ylim(0.0,1)
plt.xlabel(r"$y$", fontsize = 20)
plt.ylabel(r"$z$", fontsize = 20)
plt.tight_layout()

plt.savefig("figure2.png", dpi=100)	
plt.show()
####################################




