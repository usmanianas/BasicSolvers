

import numpy as np
from numpy import *



def delX(F):
	
	Nx, Ny, Nz = (F.shape[0], F.shape[1], F.shape[2])

	dX_dx = np.zeros([Nx, Ny, Nz])
	dX_dy = np.zeros([Nx, Ny, Nz])
	dX_dz = np.zeros([Nx, Ny, Nz])	

	dx = 1.0/(Nx-1.0)	
	dy = 1.0/(Ny-1.0)		
	dz = 1.0/(Nz-1.0)				
	
	dX_dx[1:Nx-1,1:Ny-1,1:Nz-1] = (F[2:Nx,1:Ny-1,1:Nz-1]-F[0:Nx-2,1:Ny-1,1:Nz-1])/(2.0*dx)
	dX_dy[1:Nx-1,1:Ny-1,1:Nz-1] = (F[1:Nx-1,2:Ny,1:Nz-1]-F[1:Nx-1,0:Ny-2,1:Nz-1])/(2.0*dy)
	dX_dz[1:Nx-1,1:Ny-1,1:Nz-1] = (F[1:Nx-1,1:Ny-1,2:Nz]-F[1:Nx-1,1:Ny-1,0:Nz-2])/(2.0*dz)

	'''
	for i in range(1,Nx-1):
		for j in range(1,Ny-1):
			for k in range(1,Nz-1):
				dX_dx[i,j,k] = (F[i+1,j,k]-F[i-1,j,k])/(2.0*dx)
				dX_dy[i,j,k] = (F[i,j+1,k]-F[i,j-1,k])/(2.0*dy)
				dX_dz[i,j,k] = (F[i,j,k+1]-F[i,j,k-1])/(2.0*dz)
	'''			
				
	return dX_dx, dX_dy, dX_dz
	
def del2X(F):
	
	Nx, Ny, Nz = (F.shape[0], F.shape[1], F.shape[2])
	
	d2X_dx2 = np.zeros([Nx, Ny, Nz])
	d2X_dy2 = np.zeros([Nx, Ny, Nz])
	d2X_dz2 = np.zeros([Nx, Ny, Nz])	
	
	dx = 1.0/(Nx-1.0)	
	dy = 1.0/(Ny-1.0)		
	dz = 1.0/(Nz-1.0)

	d2X_dx2[1:Nx-1,1:Ny-1,1:Nz-1] = (F[2:Nx,1:Ny-1,1:Nz-1]-2.0*F[1:Nx-1,1:Ny-1,1:Nz-1]+F[0:Nx-2,1:Ny-1,1:Nz-1])/(dx*dx)
	d2X_dy2[1:Nx-1,1:Ny-1,1:Nz-1] = (F[1:Nx-1,2:Ny,1:Nz-1]-2.0*F[1:Nx-1,1:Ny-1,1:Nz-1]+F[1:Nx-1,0:Ny-2,1:Nz-1])/(dy*dy)
	d2X_dz2[1:Nx-1,1:Ny-1,1:Nz-1] = (F[1:Nx-1,1:Ny-1,2:Nz]-2.0*F[1:Nx-1,1:Ny-1,1:Nz-1]+F[1:Nx-1,1:Ny-1,0:Nz-2])/(dz*dz)

	'''
	for i in range(1,Nx-1):
		for j in range(1,Ny-1):
			for k in range(1,Nz-1):
				d2X_dx2[i,j,k] = (F[i+1,j,k]-2.0*F[i,j,k]+F[i-1,j,k])/(dx*dx)
				d2X_dy2[i,j,k] = (F[i,j+1,k]-2.0*F[i,j,k]+F[i,j-1,k])/(dy*dy)
				d2X_dz2[i,j,k] = (F[i,j,k+1]-2.0*F[i,j,k]+F[i,j,k-1])/(dz*dz)
	'''			
	return d2X_dx2, d2X_dy2, d2X_dz2		






