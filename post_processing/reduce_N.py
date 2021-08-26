
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

print (Vx.shape)
print (np.mean(T))

Nx = P.shape[0]
Ny = P.shape[1]
Nz = P.shape[2]

n = 8   #2, 4, 8, 16

Vx = Vx[0:Nx:n,0:Ny:n,0:Nz:n]
Vy = Vy[0:Nx:n,0:Ny:n,0:Nz:n]
Vz = Vz[0:Nx:n,0:Ny:n,0:Nz:n]
T = T[0:Nx:n,0:Ny:n,0:Nz:n]
P = P[0:Nx:n,0:Ny:n,0:Nz:n]

print (np.mean(T))

hf = h5py.File('reduced_N.h5', 'w')
hf.create_dataset('Vx', data=Vx)
hf.create_dataset('Vy', data=Vy)
hf.create_dataset('Vz', data=Vz)
hf.create_dataset('T', data=T)
hf.create_dataset('P', data=P)
hf.close()

print (Vx.shape)






	
	
	


	
	


