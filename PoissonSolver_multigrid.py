
import numpy as np
import cupy as cp
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
from datetime import datetime
import random 
import numba


def initMG(gn1, gn2, gn3, vDepth, Target):
    global N, Nx, Ny, Nz
    global pData, rData, sData, iTemp
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor, maxCount, nList, VDepth
    global mghx2, mghy2, mghz2, mghx, mghy, mghz

    Lx, Ly, Lz = 1, 1, 1

    sInd = np.array([gn1, gn2, gn3])

    VDepth = vDepth

    if VDepth > min(sInd)-1: 
        print("multigrid exceeds the depth limit")
        print("Maximum Possible depth=", min(sInd)-1)
        print("Setting new depth=", min(sInd)-1)
        VDepth = min(sInd)-1

    # N should be of the form 2^n
    # Then there will be 2^n + 2 points, including two ghost points
    sLst = [2**x for x in range(12)]

    Nx, Ny, Nz = sLst[sInd[0]], sLst[sInd[1]], sLst[sInd[2]]


    #############################################################


    # Get array of grid sizes are tuples corresponding to each level of V-Cycle
    N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

    n = N

    # Define array of grid spacings along X
    h0 = Lx/(N[0][0])
    mghx = [h0*(2**x) for x in range(VDepth+1)]

    # Define array of grid spacings along Y
    h0 = Ly/(N[0][1])
    mghy = [h0*(2**x) for x in range(VDepth+1)]

    # Define array of grid spacings along Z
    h0 = Lz/(N[0][2])
    mghz = [h0*(2**x) for x in range(VDepth+1)]

    # Square of hx, used in finite difference formulae
    mghx2 = [x*x for x in mghx]

    # Square of hy, used in finite difference formulae
    mghy2 = [x*x for x in mghy]

    # Square of hz, used in finite difference formulae
    mghz2 = [x*x for x in mghz]

    # Cross product of hy and hz, used in finite difference formulae
    hyhz = [mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Cross product of hx and hz, used in finite difference formulae
    hzhx = [mghx2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Cross product of hx and hy, used in finite difference formulae
    hxhy = [mghx2[i]*mghy2[i] for i in range(VDepth + 1)]

    # Cross product of hx, hy and hz used in finite difference formulae
    hxhyhz = [mghx2[i]*mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Factor in denominator of Gauss-Seidel iterations
    gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i])) for i in range(VDepth + 1)]

    # Maximum number of iterations while solving at coarsest level
    maxCount = 10*N[-1][0]*N[-1][1]*N[-1][2]

    # Integer specifying the level of V-cycle at any point while solving
    vLev = 0

    nList = np.array(N)

    if Target=="CPU":
        pData = [np.zeros(tuple(x)) for x in nList + 2]
        rData = [np.zeros_like(x) for x in pData]
        sData = [np.zeros_like(x) for x in pData]
        iTemp = [np.zeros_like(x) for x in pData]

    if Target=="GPU":
        pData = [cp.zeros(tuple(x)) for x in nList + 2]
        rData = [cp.zeros_like(x) for x in pData]
        sData = [cp.zeros_like(x) for x in pData]
        iTemp = [cp.zeros_like(x) for x in pData]

############################## MULTI-GRID SOLVER ###############################

# The root function of MG-solver. And H is the RHS
def Poisson_MG(H, preSm, pstSm, MaxvcCnt, tolerance, gssor):
    global N, Nx, Ny, Nz
    global pData, rData, sData, iTemp
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor, maxCount, nList, VDepth
    global mghx2, mghy2, mghz2, mghx, mghy, mghz

    #initMG(gn1, gn2, gn3, Target)

    print(Nx, Ny, Nz)
    print(gssor)

    rData[0] = H

    vcnt = 0
    t1 = datetime.now()
    for i in range(MaxvcCnt):
        v_cycle(preSm, pstSm, tolerance, gssor)

        resVal = float(np.amax(np.abs(H[1:-1, 1:-1, 1:-1] - laplace(pData[0]))))

        vcnt += 1

        print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

        if resVal < tolerance:
            #print("multigrid v-cycles:", vcnt)
            break

    t2 = datetime.now()
    print("Execution time:",t2-t1)


    return pData[0]


# Multigrid V-cycle without the use of recursion
def v_cycle(preSm, pstSm, tolerance, gssor):
    global N, Nx, Ny, Nz
    global pData, rData, sData, iTemp
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor, maxCount, nList, VDepth
    global mghx2, mghy2, mghz2, mghx, mghy, mghz
    global VDepth
    global vLev
    #global pstSm, preSm

    vLev = 0

    # Pre-smoothing
    smooth(pData, rData, preSm, gssor)

    zeroBC = True
    for i in range(VDepth):
        # Compute residual
        calcResidual()

        # Copy smoothed pressure for later use
        sData[vLev] = np.copy(pData[vLev])

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing!
        if vLev == VDepth:
            #solve(tolerance)
            smooth(pData, rData, preSm, gssor)
        else:
            smooth(pData, rData, preSm, gssor)

    # Prolongation operations
    for i in range(VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Post-smoothing
        smooth(pData, rData, pstSm, gssor)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(pData, rData, sCount, gssor):
    
    for iCnt in range(sCount):
        #imposePpBCs(pData[vLev])

        pData[vLev][0, :, :], pData[vLev][-1, :, :] = -pData[vLev][1, :, :], -pData[vLev][-2, :, :]
        pData[vLev][:, 0, :], pData[vLev][:, -1, :] = -pData[vLev][:, 1, :], -pData[vLev][:, -2, :]
        pData[vLev][:, :, 0], pData[vLev][:, :, -1] = -pData[vLev][:, :, 1], -pData[vLev][:, :, -2]

        for i in range(1, N[vLev][0]+1):
            for j in range(1, N[vLev][1]+1):
                for k in range(1, N[vLev][2]+1):
                    pData[vLev][i, j, k] = (1-gssor)*pData[vLev][i, j, k] + gssor*(hyhz[vLev]*(pData[vLev][i+1, j, k] + pData[vLev][i-1, j, k]) +
                                                   hzhx[vLev]*(pData[vLev][i, j+1, k] + pData[vLev][i, j-1, k]) +
                                                   hxhy[vLev]*(pData[vLev][i, j, k+1] + pData[vLev][i, j, k-1]) -
                                                 hxhyhz[vLev]*rData[vLev][i, j, k]) * gsFactor[vLev]

        '''
        # Jacobi    
        pData[vLev][1:-1, 1:-1, 1:-1] = (hyhz[vLev]*(pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                                   hzhx[vLev]*(pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                                   hxhy[vLev]*(pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) -
                                                 hxhyhz[vLev]*rData[vLev][1:-1, 1:-1, 1:-1]) * gsFactor[vLev]
        '''
        
        
    #imposePpBCs(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1, 1:-1] = rData[vLev][1:-1, 1:-1, 1:-1] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    rData[vLev][1:-1, 1:-1, 1:-1] = (iTemp[pLev][1:-1:2, 1:-1:2, 1:-1:2] + iTemp[pLev][2::2, 2::2, 2::2] +
                                     iTemp[pLev][1:-1:2, 1:-1:2, 2::2] + iTemp[pLev][2::2, 2::2, 1:-1:2] +
                                     iTemp[pLev][1:-1:2, 2::2, 1:-1:2] + iTemp[pLev][2::2, 1:-1:2, 2::2] +
                                     iTemp[pLev][2::2, 1:-1:2, 1:-1:2] + iTemp[pLev][1:-1:2, 2::2, 2::2])/8


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve(tolerance):
    global N, vLev
    global gsFactor
    global maxCount
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]

    jCnt = 0
    while True:
        imposePpBCs(pData[vLev])

        pData[vLev][1:-1, 1:-1, 1:-1] = (hyhz[vLev]*(pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                                   hzhx[vLev]*(pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                                   hxhy[vLev]*(pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) -
                                                 hxhyhz[vLev]*rData[vLev][1:-1, 1:-1, 1:-1]) * gsFactor[vLev]
            
        maxErr = np.amax(np.abs(rData[vLev][1:-1, 1:-1, 1:-1] - laplace(pData[vLev])))

        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging at the coarsest level. Aborting")
            quit()

    imposePpBCs(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = pData[vLev][2::2, 1:-1:2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 1:-1:2, 2::2] = \
    pData[vLev][2::2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 2::2] = pData[vLev][2::2, 1:-1:2, 2::2] = pData[vLev][2::2, 2::2, 2::2] = pData[pLev][1:-1, 1:-1, 1:-1]


# Computes the 3D laplacian of function
def laplace(function):

    laplacian = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1])/mghx2[vLev] + 
                 (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1])/mghy2[vLev] +
                 (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:])/mghz2[vLev])

    return laplacian

def imposePpBCs(Pp):
    Pp[0, :, :], Pp[-1, :, :] = -Pp[1, :, :], -Pp[-2, :, :]
    Pp[:, 0, :], Pp[:, -1, :] = -Pp[:, 1, :], -Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = -Pp[:, :, 1], -Pp[:, :, -2]

Nx, Ny, Nz = 32, 32, 32
rhs = np.random.rand(Nx+2, Ny+2, Nz+2)
initMG(gn1=5, gn2=5, gn3=5, vDepth=3, Target="CPU")
pData[0] = Poisson_MG(rhs, preSm = 5, pstSm = 5, MaxvcCnt = 100, tolerance = 1e-5, gssor = 1.4)