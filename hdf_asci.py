import h5py
import numpy as np
import pylab as plt

def hdf5_reader(filename,dataset):
    file_V1_read = h5py.File(filename)
    dataset_V1_read = file_V1_read["/"+dataset]
    V1=dataset_V1_read[:,:,:]
    return V1
    
filename = "Soln_100.00000.h5"

Vx = hdf5_reader(filename, "Vx")
Vy = hdf5_reader(filename, "Vy")
Vz = hdf5_reader(filename, "Vz")
T = hdf5_reader(filename, "T")
P = hdf5_reader(filename, "P")

Nx, Ny, Nz = P.shape[0], P.shape[1], P.shape[2]

hx, hy, hz = 1/(Nx), 1/(Ny), 1/(Nz)

x = np.linspace(0, 1 + hx, Nx + 2, endpoint=True) - hx/2
y = np.linspace(0, 1 + hx, Ny + 2, endpoint=True) - hy/2
z = np.linspace(0, 1 + hx, Nz + 2, endpoint=True) - hz/2

betax, betay, betaz = 1.0e-10, 1.0e-10, 1.0e-10

x = 0.5*(1.0 - np.tanh(betax*(1.0 - 2*x))/np.tanh(betax))
y = 0.5*(1.0 - np.tanh(betay*(1.0 - 2*y))/np.tanh(betay))
z = 0.5*(1.0 - np.tanh(betaz*(1.0 - 2*z))/np.tanh(betaz))	

print("Grid", Nx, Ny, Nz)
print("beta", betax, betay, betaz)

f = open('Soln.dat', 'w+') 
f.write('VARIABLES = "X", "Y", "Z", "Vx", "Vy", "Vz", "P", "T" \n')
f.write("ZONE I = %d" %(Nx))
f.write("  J = %d" %(Ny))
f.write("  K = %d" %(Nz))
f.write("  DATAPACKING=POINT \n")
print("Writing ascii file")
for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):
			f.write("%.15e	 %.15e	 %.15e	 %.15e	%.15e	%.15e	%.15e	%.15e \n" % (x[i],y[j],z[k],Vx[i,j,k],Vy[i,j,k],Vz[i,j,k],P[i,j,k],T[i,j,k]))
			
f.close()
print("Done!")
	
	
	


	
	


