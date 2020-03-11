import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import cmath
import time
from mpl_toolkits import mplot3d

import nbimporter
import Fun_OL

########################## Graphics for OPTICAL LATTICE ##########################

#parameters
J = 1
U = 0
V0 = 10**-5
Nx = 1000
Ny = 2
case = ['SC','Harmonic','Isotropic']
temporary = '%s_%s_%s_Nx_%.1iNy_%.1i_J_%.2f_U_%.4f_V_%.4f' %\
		 (case[0],case[1],case[2],Nx,Ny,J,U,V0)

data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)

system = data[0]
E0 = data[1]
n0 = data[2]
E0_all = data[3]
H_KV = data[4]
eigVal,eigVectors = linalg.eigh(H_KV)

#n0 = Fun_OL.vector2matrix(n0,Nx,Ny)

fig_OL = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

ax1.set_xlabel('Iterations')
ax1.set_ylabel('$E$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
pyplot.suptitle('%s, %s,%s, Nx = %.1i, Ny = %.1i, J = %.2f, U = %.4f, V = %.4f' %\
		 (case[0],case[1],case[2],Nx,Ny,J,U,V0))

if Ny>2 and Nx>2:
	ax2.set_title('Density')
	pyplot.pcolormesh(n0)
	pyplot.colorbar(ax=ax2)
	ax2.set_xlabel('$x$')
	ax2.set_ylabel('$y$')

if Ny<=2 or Nx<=2:
	pyplot.plot(np.arange(Nx),np.sqrt(n0[0:Nx]))
	ax2.set_xlabel('$x$')
	ax2.set_ylabel('$\psi$')

#ax2.plot(np.arange(Nx*Ny),eigVal,'.')
ax1.plot(E0_all,'.')
#ax2.plot(E0,'bo')
pyplot.show()

temp = '%s_%s_%s_Nx_%.1iNy_%.1i_J_%.3f_U_%.4f_V_%.4f' %\
		 (case[0],case[1],case[2],Nx,Ny,J,U,V0)
fig_OL.savefig("Figures_OL/fig_"+temp+".pdf")