
import numpy as np
from numpy import *
import pylab as plt
import matplotlib as mpl

mpl.style.use('classic')

plt.rcParams['xtick.major.size'] = 5

plt.rcParams['xtick.major.width'] = 1

#plt.rcParams['xtick.minor.size'] = 5

#plt.rcParams['xtick.minor.width'] = 1

plt.rcParams['ytick.major.size'] = 5

plt.rcParams['ytick.major.width'] = 1

#plt.rcParams['ytick.minor.size'] = 5

#plt.rcParams['ytick.minor.width'] = 1


plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
font = {'family' : 'serif', 'weight' : 'normal', 'size' : 20}
plt.rc('font', **font)



def yz_plot(Fy,Fz,ns,xp):
	
	Nx, Ny, Nz = (Fy.shape[0], Fy.shape[1], Fy.shape[2])
	
	#print Fy.shape
	
	FyNew = zeros([Ny, Nz])
	FzNew = zeros([Ny, Nz])
	
	for j in range(0,Ny):
			for k in range(0,Nz):
				FyNew[j,k] = Fy[xp,j,k]			
				FzNew[j,k] = Fz[xp,j,k]	
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)			
	
	print ("vector plot at x =", x[xp])
	
	Z,Y = np.meshgrid(y,z)
	
	plt.figure(1)
	#plt.title(r"$V_y\hat{e}_y+V_z\hat{e}_z$", fontsize = 20)
	#plt.title(r"$-(\partial p/\partial y)\hat{e}_y-(\partial p/\partial z)\hat{e}_z$", fontsize = 20)
	#plt.title(r"$Fc_y\hat{e}_y+Fc_z\hat{e}_z$", fontsize = 20)
	#plt.title(r"$\nu\nabla^2V_y\hat{e}_y+\nu\nabla^2V_z\hat{e}_z$", fontsize = 20)
		
	plt.quiver(Y[0:Nx:ns,0:Ny:ns],Z[0:Nx:ns,0:Ny:ns],FyNew[0:Nx:ns,0:Ny:ns],FzNew[0:Nx:ns,0:Ny:ns])
	
	plt.xlim(0.0,1)
	plt.ylim(0.0,1)
	
	#plt.xticks((0,0.5,1))
	
	plt.xlabel(r"$y$", fontsize = 25)
	plt.ylabel(r"$z$", fontsize = 25)
	plt.axis('scaled')
	plt.tight_layout()	
	#plt.savefig("VyVz_cs_x0p02.png")	
	#plt.savefig("-gradP_x0p05.svg")
	#plt.savefig("Fc_x0p05.svg")
	plt.savefig("del2V_x0p05.svg")
	

def xz_plot(Fx,Fz,ns,yp):
	
	Nx, Ny, Nz = (Fx.shape[0], Fx.shape[1], Fx.shape[2])
	
	FxNew = zeros([Ny, Nz])
	FzNew = zeros([Ny, Nz])
	
	for i in range(0,Nx):
			for k in range(0,Nz):
				FxNew[i,k] = Fx[i,yp,k]
				FzNew[i,k] = Fz[i,yp,k]	
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)			

	print ("vector plot at y =", y[yp])
	
	Z,X = np.meshgrid(x,z)
	
	plt.figure(2)
	plt.title(r"$V_x\hat{e}_x+V_z\hat{e}_z$", fontsize = 20)	
	
	plt.quiver(X[0:Nx:ns,0:Ny:ns],Z[0:Nx:ns,0:Ny:ns],FxNew[0:Nx:ns,0:Ny:ns],FzNew[0:Nx:ns,0:Ny:ns])
	
	plt.xlim(0.0,1)
	plt.ylim(0.0,1)
	
	#plt.xticks((0,0.5,1))
	
	plt.xlabel(r"$x$", fontsize = 20)
	plt.ylabel(r"$z$", fontsize = 20)
	plt.axis('scaled')
	plt.tight_layout()	
	#plt.savefig("VxVz_cs_y0p5.png")

	
	
def xy_plot(Fx,Fy,ns,zp):
	
	Nx, Ny, Nz = (Fx.shape[0], Fx.shape[1], Fx.shape[2])
	
	FxNew = zeros([Ny, Nz])
	FyNew = zeros([Ny, Nz])
	
	for i in range(0,Nx):
			for j in range(0,Ny):
				FxNew[i,j] = Fx[i,j,zp]
				FyNew[i,j] = Fy[i,j,zp]	
	
	x = np.linspace(0, 1, Nx, endpoint=True)		
	y = np.linspace(0, 1, Ny, endpoint=True)
	z = np.linspace(0, 1, Nz, endpoint=True)			

	print ("vector plot at z =", z[zp])
	
	Y,X = np.meshgrid(x,y)
	
	plt.figure(3)
	plt.title(r"$V_x\hat{e}_x+V_y\hat{e}_y$", fontsize = 20)	
	
	plt.quiver(X[0:Nx:ns,0:Ny:ns],Y[0:Nx:ns,0:Ny:ns],FxNew[0:Nx:ns,0:Ny:ns],FyNew[0:Nx:ns,0:Ny:ns])
	
	plt.xlim(0.0,1)
	plt.ylim(0.0,1)
	
	#plt.xticks((0,0.5,1))
	
	plt.xlabel(r"$x$", fontsize = 20)
	plt.ylabel(r"$y$", fontsize = 20)
	plt.axis('scaled')
	plt.tight_layout()		
	#plt.savefig("VxVy_cs_z0p5.png")

			
plt.show()	





