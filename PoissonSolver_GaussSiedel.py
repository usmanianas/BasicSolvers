
import numpy as np
import cupy as cp
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
from datetime import datetime
import random 
import numba

Target = "CPU"   # CPU or GPU

Lx, Ly, Lz = 1, 1, 1

Nx, Ny, Nz = 16, 16, 16

hx, hy, hz = Lx/(Nx), Ly/(Ny), Lz/(Nz)

i2hx, i2hy, i2hz = 1/(2*hx), 1/(2*hy), 1/(2*hz)

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2


if Target=="CPU":
    Pp = np.zeros([Nx+2, Ny+2, Nz+2])
    rhs = np.ones([Nx+2, Ny+2, Nz+2]) #np.random.rand(Nx+2, Ny+2, Nz+2)
    tmp = np.zeros([Nx+2, Ny+2, Nz+2])

    
if Target=="GPU":
    Pp = np.zeros([Nx+2, Ny+2, Nz+2])
    rhs = np.random.rand(Nx+2, Ny+2, Nz+2)
    tmp = np.zeros([Nx+2, Ny+2, Nz+2])
    

def Poisson_Jacobi(Pp, rho): 
    PoissonTolerance = 1e-5  
    jCnt = 0   
    Pp.fill(0.0)
    while True:
            
        Pp[1:-1, 1:-1, 1:-1] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[1:-1, 1:-1, 1:-1] - 
                                           idx2*(Pp[:-2, 1:-1, 1:-1] + Pp[2:, 1:-1, 1:-1]) -
                                           idy2*(Pp[1:-1, :-2, 1:-1] + Pp[1:-1, 2:, 1:-1]) -
                                           idz2*(Pp[1:-1, 1:-1, :-2] + Pp[1:-1, 1:-1, 2:]))   

        Pp[0, :, :], Pp[-1, :, :] = -Pp[1, :, :], -Pp[-2, :, :]
        Pp[:, 0, :], Pp[:, -1, :] = -Pp[:, 1, :], -Pp[:, -2, :]
        Pp[:, :, 0], Pp[:, :, -1] = -Pp[:, :, 1], -Pp[:, :, -2] 
            
        maxErr = np.amax(np.abs(rho[1:-1, 1:-1, 1:-1] -((
                            (Pp[:-2, 1:-1, 1:-1] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[2:, 1:-1, 1:-1])*idx2 +
                            (Pp[1:-1, :-2, 1:-1] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[1:-1, 2:, 1:-1])*idy2 +
                            (Pp[1:-1, 1:-1, :-2] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[1:-1, 1:-1, 2:])*idz2))))

        if (jCnt % 100 == 0): print(jCnt, maxErr)

        jCnt += 1
        
        if maxErr < PoissonTolerance:
            print(jCnt)
            break
        
        if jCnt > 10000:#maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
        
    return Pp  
    
@numba.njit
def Poisson_GS(Pp, rho, tmp, gssor):   
    PoissonTolerance = 1e-5
    jCnt = 0   
    Pp.fill(0.0)
    while True:

        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                for k in range(1, Nz+1):
                    Pp[i, j, k] = (1-gssor)*Pp[i, j, k] + gssor*(1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                           idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                           idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                           idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))   
            
        Pp[0, :, :], Pp[-1, :, :] = -Pp[1, :, :], -Pp[-2, :, :]
        Pp[:, 0, :], Pp[:, -1, :] = -Pp[:, 1, :], -Pp[:, -2, :]
        Pp[:, :, 0], Pp[:, :, -1] = -Pp[:, :, 1], -Pp[:, :, -2]  
            
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                for k in range(1, Nz+1):
                    tmp[i, j, k] = rho[i, j, k] -((
                            (Pp[i+1, j, k] - 2.0*Pp[i, j, k] + Pp[i-1, j, k])*idx2 +
                            (Pp[i, j+1, k] - 2.0*Pp[i, j, k] + Pp[i, j-1, k])*idy2 +
                            (Pp[i, j, k+1] - 2.0*Pp[i, j, k] + Pp[i, j, k-1])*idz2))
            
        maxErr = np.amax(np.abs(tmp[1:-1, 1:-1, 1:-1]))

        if (jCnt % 100 == 0): print(jCnt, maxErr)

        jCnt += 1
        
        if maxErr < PoissonTolerance:
            print(jCnt)
            break
        
        if jCnt > 10000:#maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            break
        
    return Pp     

    
def Poisson_RBGS(Pp, rho):   
    PoissonTolerance = 1e-5
    jCnt = 0   
    Pp.fill(0.0)
    while True:

        Pp[1:-1:2, 1:-1:2, 1:-1:2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[1:-1:2, 1:-1:2, 1:-1:2] - idx2*(Pp[2::2, 1:-1:2, 1:-1:2] + Pp[:-2:2, 1:-1:2, 1:-1:2]) -
                                                   idy2*(Pp[1:-1:2, 2::2, 1:-1:2] + Pp[1:-1:2, :-2:2, 1:-1:2]) -
                                                   idz2*(Pp[1:-1:2, 1:-1:2, 2::2] + Pp[1:-1:2, 1:-1:2, :-2:2]))

            # 1, 1, 0 configuration
        Pp[2::2, 2::2, 1:-1:2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[2::2, 2::2, 1:-1:2] - idx2*(Pp[3::2, 2::2, 1:-1:2] + Pp[1:-1:2, 2::2, 1:-1:2]) -
                                               idy2*(Pp[2::2, 3::2, 1:-1:2] + Pp[2::2, 1:-1:2, 1:-1:2]) -
                                               idz2*(Pp[2::2, 2::2, 2::2] + Pp[2::2, 2::2, :-2:2]))

            # 1, 0, 1 configuration
        Pp[2::2, 1:-1:2, 2::2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[2::2, 1:-1:2, 2::2] - idx2*(Pp[3::2, 1:-1:2, 2::2] + Pp[1:-1:2, 1:-1:2, 2::2]) -
                                               idy2*(Pp[2::2, 2::2, 2::2] + Pp[2::2, :-2:2, 2::2]) -
                                               idz2*(Pp[2::2, 1:-1:2, 3::2] + Pp[2::2, 1:-1:2, 1:-1:2]))

            # 0, 1, 1 configuration
        Pp[1:-1:2, 2::2, 2::2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[1:-1:2, 2::2, 2::2] - idx2*(Pp[2::2, 2::2, 2::2] + Pp[:-2:2, 2::2, 2::2]) -
                                               idy2*(Pp[1:-1:2, 3::2, 2::2] + Pp[1:-1:2, 1:-1:2, 2::2]) -
                                               idz2*(Pp[1:-1:2, 2::2, 3::2] + Pp[1:-1:2, 2::2, 1:-1:2]))

            # Update black cells
            # 1, 0, 0 configuration
        Pp[2::2, 1:-1:2, 1:-1:2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[2::2, 1:-1:2, 1:-1:2] - idx2*(Pp[3::2, 1:-1:2, 1:-1:2] + Pp[1:-1:2, 1:-1:2, 1:-1:2]) -
                                                 idy2*(Pp[2::2, 2::2, 1:-1:2] + Pp[2::2, :-2:2, 1:-1:2]) -
                                                 idz2*(Pp[2::2, 1:-1:2, 2::2] + Pp[2::2, 1:-1:2, :-2:2]))

            # 0, 1, 0 configuration
        Pp[1:-1:2, 2::2, 1:-1:2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[1:-1:2, 2::2, 1:-1:2] - idx2*(Pp[2::2, 2::2, 1:-1:2] + Pp[:-2:2, 2::2, 1:-1:2]) -
                                                 idy2*(Pp[1:-1:2, 3::2, 1:-1:2] + Pp[1:-1:2, 1:-1:2, 1:-1:2]) -
                                                 idz2*(Pp[1:-1:2, 2::2, 2::2] + Pp[1:-1:2, 2::2, :-2:2]))
            # 0, 0, 1 configuration
        Pp[1:-1:2, 1:-1:2, 2::2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[1:-1:2, 1:-1:2, 2::2] - idx2*(Pp[2::2, 1:-1:2, 2::2] + Pp[:-2:2, 1:-1:2, 2::2]) -
                                                 idy2*(Pp[1:-1:2, 2::2, 2::2] + Pp[1:-1:2, :-2:2, 2::2]) -
                                                 idz2*(Pp[1:-1:2, 1:-1:2, 3::2] + Pp[1:-1:2, 1:-1:2, 1:-1:2]))

            # 1, 1, 1 configuration
        Pp[2::2, 2::2, 2::2] = 1.0/(-2.0*(idx2 + idy2 + idz2))*(rho[2::2, 2::2, 2::2] - idx2*(Pp[3::2, 2::2, 2::2] + Pp[1:-1:2, 2::2, 2::2]) -
                                             idy2*(Pp[2::2, 3::2, 2::2] + Pp[2::2, 1:-1:2, 2::2]) -
                                             idz2*(Pp[2::2, 2::2, 3::2] + Pp[2::2, 2::2, 1:-1:2]))
            
        Pp[0, :, :], Pp[-1, :, :] = -Pp[1, :, :], -Pp[-2, :, :]
        Pp[:, 0, :], Pp[:, -1, :] = -Pp[:, 1, :], -Pp[:, -2, :]
        Pp[:, :, 0], Pp[:, :, -1] = -Pp[:, :, 1], -Pp[:, :, -2]  
            
        maxErr = np.amax(np.abs(rho[1:-1, 1:-1, 1:-1] -((
                            (Pp[:-2, 1:-1, 1:-1] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[2:, 1:-1, 1:-1])*idx2 +
                            (Pp[1:-1, :-2, 1:-1] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[1:-1, 2:, 1:-1])*idy2 +
                            (Pp[1:-1, 1:-1, :-2] - 2.0*Pp[1:-1, 1:-1, 1:-1] + Pp[1:-1, 1:-1, 2:])*idz2))))


        if (jCnt % 100 == 0): print(jCnt, maxErr)

        jCnt += 1
        
        if maxErr < PoissonTolerance:
            print(jCnt)
            break
        
        if jCnt > 10000:#maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            break
        
    return Pp     

     
#Pp = Poisson_Jacobi(Pp, rhs)
#Pp = Poisson_GS(Pp, rhs, tmp, gssor = 1.0)
Pp = Poisson_RBGS(Pp, rhs)
