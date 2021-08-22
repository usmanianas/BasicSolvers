from numba import vectorize
import numpy as np
import cupy as cp
from datetime import datetime

cp.cuda.Device(1).use()

Nx = Ny = Nz = 64

hx, hy, hz = 1.0/(Nx-1), 1.0/(Ny-1), 1.0/(Nz-1)

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

itermax = 10000
otpt = 100
Tolerance = 1e-4

print(Nx, Ny, Nz)

Pp = cp.zeros([Nx, Ny, Nz])
rho = cp.ones([Nx, Ny, Nz]) #cp.random.rand(Nx, Ny, Nz)
tmp = cp.zeros([Nx, Ny, Nz])

jfactor = (1.0/(-2.0*(idx2 + idy2 + idz2)))

@cp.fuse()
def fuse1(rho, Pp, Ppxp, Ppxm, Ppyp, Ppym, Ppzp, Ppzm):
    return rho -(((Ppxp - 2.0*Pp + Ppxm)/hx2 +(Ppyp - 2.0*Pp + Ppym)/hy2 + (Ppzp - 2.0*Pp + Ppzm)/hz2))

@cp.fuse()
def fuse2(rho, Pppxp, Pppxm, Pppyp, Pppym, Pppzp, Pppzm):
    return jfactor * (rho - idx2*(Pppxp + Pppxm) - idy2*(Pppyp + Pppym) - idz2*(Pppzp + Pppzm))

def JacobiSolver(Pp, rho):
    for iteration in range(itermax+1):
        
        tmp[1:Nx-1, 1:Nx-1, 1:Nx-1] = fuse1(rho[1:Nx-1, 1:Nx-1, 1:Nx-1], Pp[1:Nx-1, 1:Ny-1, 1:Nz-1], Pp[2:Nx, 1:Nx-1, 1:Nx-1], Pp[0:Nx-2, 1:Ny-1, 1:Nz-1], Pp[1:Nx-1, 2:Nx, 1:Nx-1],
                        Pp[1:Nx-1, 0:Ny-2, 1:Nz-1], Pp[1:Nx-1, 1:Nx-1, 2:Nx], Pp[1:Nx-1, 1:Ny-1, 0:Nz-2])

        maxErr = cp.amax(cp.abs(tmp))

        Pp[1:Nx-1, 1:Nx-1, 1:Nx-1] = fuse2(rho[1:Nx-1, 1:Nx-1, 1:Nx-1], Pp[2:Nx, 1:Nx-1, 1:Nx-1], Pp[0:Nx-2, 1:Ny-1, 1:Nz-1], Pp[1:Nx-1, 2:Nx, 1:Nx-1],
                        Pp[1:Nx-1, 0:Ny-2, 1:Nz-1], Pp[1:Nx-1, 1:Nx-1, 2:Nx], Pp[1:Nx-1, 1:Ny-1, 0:Nz-2])

        if (iteration % otpt == 0): print(iteration, maxErr)

        iteration = iteration + 1

        if maxErr < Tolerance:
            print(iteration)
            break

    return Pp


t1 = datetime.now()

JacobiSolver(Pp, rho)

t2 = datetime.now()

print("Simulation time", t2-t1)
