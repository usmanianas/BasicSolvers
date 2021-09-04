## Calculation of Reynolds Number###

import numpy as np
from numpy import*
import pylab as plt
import matplotlib as mpl
from math import*

mpl.style.use('classic')

font = {'family' : 'serif', 'weight' : 'normal', 'size' : 20}
plt.rc('font', **font)

Ra = array([1.0e5*i for i in range(1,10**5)])
Ta = array([1.0e6*i for i in range(1,10**5)])

Pr = 1
Ra1 = 1.0e7
Ro = (Ra1/(Ta*Pr))**0.5
#print Ra

Nu = 0.167*Ra**(2.0/7.0) #*Pr**(0.20)

#print Nu


if Pr > 3: 
	delta_u = 0.5*Nu**(-1)*Pr**(1.0/3.0)*0.982**(-1.0)
	delta_t = 0.982*Pr**(-1/3)*delta_u

if 3e-4 < Pr <= 3: 
	delta_u = 0.5*Nu**(-1)*Pr**(0.357-0.022*log(Pr))
	delta_t = Pr**(-0.357+0.022*log(Pr))*delta_u


if Pr <= 1: h_bulk = Pr**(0.5)/(Ra**0.25*(Nu-1.0)**0.25)
if Pr > 1: h_bulk = 1.0/(Ra**0.25*(Nu-1.0)**0.25)

if 3e-4<Pr<=1: BL_min = 2**(-1.5)*(0.482**(-1.0))*Nu**(-1.5)*Pr**(0.5355-0.033*log(Pr))
if 1<Pr<=3: BL_min = 2**(-1.5)*(0.482**(-1.0))*Nu**(-1.5)*Pr**(0.0355-0.033*log(Pr))
if Pr>3: BL_min = 2**(-1.5)*(0.482**(-1.0))*(0.982)**(-1.5)*Nu**(-1.5)

h_Ek = 2.3*np.sqrt(2.0)*Ta**(-0.25)

#deltat =  5.7*Ra**(-0.33)   #1.0/(2.0*Nu)


print(Nu)

#print time1[17100], np.mean(Nu1[17100:]), np.max(Nu1)

plt.figure(1)
plt.semilogx(Ra, delta_u, lw = 2, label = r'$\delta_u$')
plt.semilogx(Ra, delta_t, lw = 2, label = r'$\delta_\theta$')
plt.semilogx(Ra, h_bulk, lw = 2, label = r'$h_{bulk}$')
plt.semilogx(Ra, BL_min, lw = 2, label = r'$BL_{min}$')

#plt.loglog(time2,E2, lw = 2, label = r'$\Omega=15$')

#plt.xlim(0.01,50)
#plt.ylim(3e-3,2)

plt.xlabel(r"$Ra$", fontsize = 30)
plt.ylabel(r"$\delta$", fontsize = 30)
plt.legend(loc = 0, fontsize = 15)
plt.tick_params(axis='both',labelsize=20)
plt.tight_layout()
plt.grid()

#plt.savefig("Etime.png")



plt.figure(2)
plt.semilogx(Ta, h_Ek, lw = 2, label = r'$h_{Ek}$')

#plt.loglog(time2,E2, lw = 2, label = r'$\Omega=15$')

#plt.xlim(0.01,50)
#plt.ylim(3e-3,2)

plt.xlabel(r"$Ta$", fontsize = 30)
plt.ylabel(r"$h_{Ek}$", fontsize = 30)
plt.legend(loc = 0, fontsize = 15)
plt.tick_params(axis='both',labelsize=20)
plt.tight_layout()
plt.grid()

#plt.savefig("Etime.png")

'''
N = array([64.0, 128.0, 256.0, 512.0, 1024.0])

deltaX = N**(-1.0)

plt.figure(3)
plt.semilogx(N, deltaX, lw = 2, label = r'$\delta x$')

#plt.loglog(time2,E2, lw = 2, label = r'$\Omega=15$')

#plt.xlim(0.01,50)
#plt.ylim(3e-3,2)

plt.xlabel(r"$N$", fontsize = 30)
plt.ylabel(r"$\delta x$", fontsize = 30)
plt.legend(loc = 0, fontsize = 15)
plt.tick_params(axis='both',labelsize=20)
plt.tight_layout()
plt.grid()
#plt.savefig("Etime.png")


plt.figure(4)
plt.semilogx(Ta,Ro, lw = 2, label = Ra1)

#plt.loglog(time2,E2, lw = 2, label = r'$\Omega=15$')

#plt.xlim(0.01,50)
plt.ylim(0.1,1)

plt.xlabel(r"$Ta$", fontsize = 30)
plt.ylabel(r"$Ro$", fontsize = 30)
plt.legend(loc = 0, fontsize = 15)
plt.tick_params(axis='both',labelsize=20)
plt.tight_layout()
plt.grid()
#plt.savefig("Etime.png")
'''
plt.show()




