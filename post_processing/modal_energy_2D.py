
import numpy as np
from numpy import *


def yz_mode(Fy,Fz,ky,kz,xp):
	
	Nx, Ny, Nz = (Fy.shape[0], Fy.shape[1], Fy.shape[2])
	
	#print Fy.shape
	
	FyNew = zeros([Ny, Nz])
	FzNew = zeros([Ny, Nz])
	
	for j in range(0,Ny):
			for k in range(0,Nz):
				FyNew[j,k] = Fy[xp,j,k]			
				FzNew[j,k] = Fz[xp,j,k]	
	
	Eyz = mean(FyNew**2+FzNew**2)			
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)	
	
	dx = x[2]-x[1]
	dy = y[2]-y[1]
	dz = z[2]-z[1]
		
	print ("modes at x =", x[xp])
	print ("m, n, %Ey, %Ez, %Ey+%Ez ")

	Z,Y = np.meshgrid(y,z)
	for m in range(ky):
		for n in range(kz):
			A1, A2, B1, B2 = (0,0,0,0)
			for j in range(0,Ny):
				for k in range(0,Nz):
					A1 = A1 + (FyNew[j,k]*sin(m*np.pi*y[j])*cos(n*np.pi*z[k])*dy*dz)
					A2 = A2 + (FzNew[j,k]*cos(m*np.pi*y[j])*sin(n*np.pi*z[k])*dy*dz)				
					B1 = B1 + ((sin(m*np.pi*y[j])**2.0)*(cos(n*np.pi*z[k])**2.0)*dy*dz)
					B2 = B2 + ((cos(m*np.pi*y[j])**2.0)*(sin(n*np.pi*z[k])**2.0)*dy*dz)
					
			if(m !=0 and n != 0):		
				Fyf = (A1/B1)*sin(m*np.pi*Y)*cos(n*np.pi*Z)
				Fzf = (A2/B2)*cos(m*np.pi*Y)*sin(n*np.pi*Z)	
			elif(m == 0 and n != 0):
					Fyf = 0
					Fzf = (A2/B2)*cos(m*np.pi*Y)*sin(n*np.pi*Z)													
			elif(m != 0 and n == 0):
					Fzf = 0
					Fyf = (A1/B1)*sin(m*np.pi*Y)*cos(n*np.pi*Z)					
			elif(m==0 and n==0):
				Fyf = 0
				Fzf = 0

			if (mean(Fyf**2+Fzf**2)/Eyz)*100 >= 1.0:
				print (m, n, (mean(Fyf**2)/Eyz)*100, (mean(Fzf**2)/Eyz)*100, (mean(Fyf**2+Fzf**2)/Eyz)*100)



			



def xz_mode(Fx,Fz,kx,kz,yp):
	
	Nx, Ny, Nz = (Fx.shape[0], Fx.shape[1], Fx.shape[2])
	
	#print Fx.shape
	
	FxNew = zeros([Nx, Nz])
	FzNew = zeros([Nx, Nz])
	
	for i in range(0,Nx):
			for k in range(0,Nz):
				FxNew[i,k] = Fx[i,yp,k]			
				FzNew[i,k] = Fz[i,yp,k]	
	
	Exz = mean(FxNew**2+FzNew**2)			
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)	
	
	dx = x[2]-x[1]
	dy = y[2]-y[1]
	dz = z[2]-z[1]
		
	print ("modes at y =", y[yp])
	print ("m, n, %Ex, %Ez, %Ex+%Ez ")
	
	Z,X = np.meshgrid(x,z)
	for m in range(kx):
		for n in range(kz):
			A1, A2, B1, B2 = (0,0,0,0)
			for i in range(0,Nx):
				for k in range(0,Nz):
					A1 = A1 + (FxNew[i,k]*sin(m*np.pi*x[i])*cos(n*np.pi*z[k])*dx*dz)
					A2 = A2 + (FzNew[i,k]*cos(m*np.pi*x[i])*sin(n*np.pi*z[k])*dx*dz)				
					B1 = B1 + ((sin(m*np.pi*x[i])**2.0)*(cos(n*np.pi*z[k])**2.0)*dx*dz)
					B2 = B2 + ((cos(m*np.pi*x[i])**2.0)*(sin(n*np.pi*z[k])**2.0)*dx*dz)
					
			if(m !=0 and n != 0):		
				Fxf = (A1/B1)*sin(m*np.pi*X)*cos(n*np.pi*Z)
				Fzf = (A2/B2)*cos(m*np.pi*X)*sin(n*np.pi*Z)	
			elif(m == 0 and n != 0):
					Fxf = 0
					Fzf = (A2/B2)*cos(m*np.pi*X)*sin(n*np.pi*Z)													
			elif(m != 0 and n == 0):
					Fzf = 0
					Fxf = (A1/B1)*sin(m*np.pi*X)*cos(n*np.pi*Z)					
			elif(m==0 and n==0):
				Fxf = 0
				Fzf = 0

			if (mean(Fxf**2+Fzf**2)/Exz)*100 >= 1:
				print (m, n, (mean(Fxf**2)/Exz)*100, (mean(Fzf**2)/Exz)*100, (mean(Fxf**2+Fzf**2)/Exz)*100)

								
			



def xy_mode(Fx,Fy,kx,ky,zp):
	
	Nx, Ny, Nz = (Fx.shape[0], Fx.shape[1], Fx.shape[2])
	
	#print Fx.shape
	
	FxNew = zeros([Nx, Ny])
	FyNew = zeros([Nx, Ny])
	
	for i in range(0,Nx):
			for j in range(0,Ny):
				FxNew[i,j] = Fx[i,j,zp]			
				FyNew[i,j] = Fy[i,j,zp]	
	
	Exy = mean(FxNew**2+FyNew**2)			
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)	
	
	dx = x[2]-x[1]
	dy = y[2]-y[1]
	dz = z[2]-z[1]
		
	print ("modes at z =", z[zp])
	print ("m, n, %Ex, %Ey, %Ex+%Ey ")
	
	Y,X = np.meshgrid(x,y)
	for m in range(kx):
		for n in range(ky):
			A1, A2, B1, B2 = (0,0,0,0)
			for i in range(0,Nx):
				for j in range(0,Ny):
					A1 = A1 + (FxNew[i,j]*sin(m*np.pi*x[i])*cos(n*np.pi*y[j])*dx*dy)
					A2 = A2 + (FyNew[i,j]*cos(m*np.pi*x[i])*sin(n*np.pi*y[j])*dx*dy)				
					B1 = B1 + ((sin(m*np.pi*x[i])**2.0)*(cos(n*np.pi*y[j])**2.0)*dx*dy)
					B2 = B2 + ((cos(m*np.pi*x[i])**2.0)*(sin(n*np.pi*y[j])**2.0)*dx*dy)
					
			if(m !=0 and n != 0):		
				Fxf = (A1/B1)*sin(m*np.pi*X)*cos(n*np.pi*Y)
				Fyf = (A2/B2)*cos(m*np.pi*X)*sin(n*np.pi*Y)	
			elif(m == 0 and n != 0):
					Fxf = 0
					Fyf = (A2/B2)*cos(m*np.pi*X)*sin(n*np.pi*Y)													
			elif(m != 0 and n == 0):
					Fyf = 0
					Fxf = (A1/B1)*sin(m*np.pi*X)*cos(n*np.pi*Y)					
			elif(m==0 and n==0):
				Fxf = 0
				Fyf = 0
			
			if 	(mean(Fxf**2+Fyf**2)/Exy)*100 >=1:
				print (m, n, (mean(Fxf**2)/Exy)*100, (mean(Fyf**2)/Exy)*100, (mean(Fxf**2+Fyf**2)/Exy)*100)

			

