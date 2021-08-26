import numpy as np
from numpy import *
import pylab as plt
import h5py
from matplotlib import cm

import Vector_Plot as vp
import derivative as der
import modal_energy_2D as mode
import matplotlib as mpl

mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1

plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
font = {'family' : 'serif', 'weight' : 'normal', 'size' : 20}
plt.rc('font', **font)
	
filename = "../Soln_100.00000.h5"	

case = 1 # 0 - Soln.dat, 1- modal energy, 2- vector plots, 3- helicity

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
T = hdf5_reader(filename, "T")

Nx, Ny, Nz = (Vx.shape[0], Vx.shape[1], Vx.shape[2])

print("Nx=",Nx, "Ny=",Ny, "Nz=",Nz)

hx, hy, hz = 1/(Nx), 1/(Ny), 1/(Nz)

x = np.linspace(0, 1 + hx, Nx + 2, endpoint=True) - hx/2
y = np.linspace(0, 1 + hx, Ny + 2, endpoint=True) - hy/2
z = np.linspace(0, 1 + hx, Nz + 2, endpoint=True) - hz/2

beta = 1.0e-10
x = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*x))/np.tanh(beta))
y = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*y))/np.tanh(beta))
z = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*z))/np.tanh(beta))	

# paraview format ###########
if ( case == 0):
	print("writing Soln.dat file")
	f = open('Soln.dat', 'w+') 
	f.write('VARIABLES = "X", "Y", "Z", "Vx", "Vy", "Vz", "P", "T" \n')
	f.write("ZONE I = %d" %(Nx))
	f.write("  J = %d" %(Ny))
	f.write("  K = %d" %(Nz))
	f.write("  DATAPACKING=POINT \n")
	
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				f.write("%.15e	 %.15e	 %.15e	 %.15e	%.15e	%.15e	%.15e	%.15e \n" % (x[i],y[j],z[k],Vx[i,j,k],Vy[i,j,k],Vz[i,j,k],P[i,j,k],T[i,j,k]))
				
	f.close()
	print("Done!")


# Derivatives ##############
if (1 == 0):
	#dP_dx, dP_dy, dP_dz = der.delX(P)
	
	d2Vx_dx2, d2Vx_dy2, d2Vx_dz2 = der.del2X(Vx)
	d2Vy_dx2, d2Vy_dy2, d2Vy_dz2 = der.del2X(Vy)
	d2Vz_dx2, d2Vz_dy2, d2Vz_dz2 = der.del2X(Vz)
	d2Vy = d2Vy_dx2 + d2Vy_dy2 + d2Vy_dz2
	d2Vz = d2Vz_dx2 + d2Vz_dy2 + d2Vz_dz2

###################################


# modal energy ####################
if (case == 1):
	mode.yz_mode(Vy, Vz, ky=3, kz=3, xp=10)
	mode.xz_mode(Vx, Vz, kx=3, kz=3, yp=10)
	mode.xy_mode(Vx, Vy, kx=3, ky=3, zp=10)

###############################



# Vector plots ###############
if (case == 2):
	vp.yz_plot(Vy, Vz, ns=10,xp=int(Nx/2))
	vp.xz_plot(Vx, Vz, ns=6, yp=int(Ny/2))
	vp.xy_plot(Vx, Vy, ns=6, zp=int(Nz/2))
	
	#vp.yz_plot(-dP_dy, -dP_dz+T, ns=10, xp= 7)
	
	#vp.yz_plot(-sqrt(Ta*Pr/Ra)*Vz, sqrt(Ta*Pr/Ra)*Vy, ns=10, xp=7)
	
	#vp.yz_plot(sqrt(Pr/Ra)*d2Vy, sqrt(Pr/Ra)*d2Vz, ns=10, xp=7)	
	
	vp.plt.show()
####################################



# Helicity ###############
if (case == 3):
	##print "Calculating Vorticity:"
	dVx_dx, dVx_dy, dVx_dz = der.delX(Vx)
	dVy_dx, dVy_dy, dVy_dz = der.delX(Vy)
	dVz_dx, dVz_dy, dVz_dz = der.delX(Vz)
	omega_x, omega_y, omega_z = (dVz_dy-dVy_dz, dVx_dz-dVz_dx, dVy_dx-dVx_dy)
	
	print("Skewness Sw = ", mean(omega_z**3.0)/(mean(omega_z**2.0))**1.5)

	print("Calculating and Plotting Helicity:")
	
	Hx, Hy, Hz = (Vx*omega_x, Vy*omega_y, Vz*omega_z)

	Hmax = np.sqrt(Vx**2.0 + Vy**2.0 + Vz**2.0) * np.sqrt(omega_x**2.0 + omega_y**2.0 + omega_z**2.0)

	Hmax[0,:,:], Hmax[Nx-1,:,:], Hmax[:,0,:], Hmax[:,Ny-1,:], Hmax[:,:,0], Hmax[:,:,Nz-1] = 1, 1, 1, 1, 1, 1

	H = Hx + Hy + Hz

	plt.figure(1)
	plt.plot(mean(Hx/Hmax, axis=(0,1)), z[1:Nx+1], lw = 2, label = r'$V_x\omega_x$')
	plt.plot(mean(Hy/Hmax, axis=(0,1)), z[1:Nx+1], lw = 2, label = r'$V_y\omega_y$')
	plt.plot(mean(Hz/Hmax, axis=(0,1)), z[1:Nx+1], lw = 2, label = r'$V_z\omega_z$')
	plt.plot(mean(H/Hmax, axis=(0,1)), z[1:Nx+1], lw = 3, label = r'$V_x\omega_x+V_y\omega_y+V_z\omega_z$')
	
	#plt.xlim(-0.15,0.15)
	#plt.ylim(1,)
	plt.axvline(x=0.0, lw=1.5, color="k", ls='--')
	plt.axhline(y=0.5, lw=1.5, color="k", ls='--')
	plt.axhline(y=0.103, lw=1.5, color="k", ls='-.')
	plt.axhline(y=1-0.103, lw=1.5, color="k", ls='-.')	
	
	#plt.xticks((-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15))
	
	plt.xlabel("$H$", fontsize = 30)
	plt.ylabel("$x$", fontsize = 30)
	plt.legend(loc = 0, fontsize = 15)
	plt.tick_params(axis='both',labelsize=20)
	plt.tight_layout()
	
	#plt.savefig("H_RC2.svg")
	plt.show()		
	

####################################






