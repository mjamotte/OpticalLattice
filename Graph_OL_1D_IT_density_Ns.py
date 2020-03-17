import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import cmath
import time
from mpl_toolkits import mplot3d

#import nbimporter
import Fun_OL

########################## Graphics for OPTICAL LATTICE ##########################

J = 1
N_s = [0,10,25,50] # N=0 actually means g = 0
V0 = 0.5*1e-3
Nx = 51
case = ['IT','Harmonic','Isotropic']

# Figure

fig_OL1D = pyplot.figure(figsize=(8,8))
ax2 = pyplot.subplot(111)
ax2.set_xlabel('$r_{center}/\sigma$')
ax2.set_ylabel('$|\psi_0|^2$')#,color='r')

for N in N_s:

	# Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_N_%1i_V_%.4f' %\
			 	(case[0],case[1],case[2],Nx,N,V0)
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data[0]
	E0_all = data[1]
	n0 = data[2]
	H_KV = data[3]

	eigVal_H_KV,eigVect_H_KV = np.linalg.eigh(H_KV)

	# Analytical gaussien 

	Trap = Fun_OL.trap_1D(system,case)
	m = 1/(2*J)
	w0 = np.sqrt(V0/m)
	sigma = np.sqrt(1/(m*w0))
	x0 = (Nx-1)/2
	positions = np.arange(Nx)
	Gauss = Fun_OL.gauss(positions,x0,sigma) # Lowest band of HKV (analytic)

	dens = abs(eigVect_H_KV[:,0])**2 # Lowest band of HKV (numeric)

	centered = 'False'
	if centered=='True':
		half_len = int(len(positions)/2)
		center = positions[half_len]
		half_pos = (positions-center)[half_len:]
		ax2.plot(half_pos/sigma,Gauss[half_len:]**2/np.sum(Gauss**2),'-g')
		ax2.plot(half_pos/sigma,dens[half_len:],'.b')
		ax2.plot(half_pos/sigma,n0[half_len:].real,'-r',label="$N = {}$".format(N))

	if centered=='False':
		ax2.plot(positions/sigma,Gauss**2/np.sum(Gauss**2),'-g')
		ax2.plot(positions/sigma,dens,'.b')
		ax2.plot(positions/sigma,n0.real,'-',label="$N = {}$".format(N))

	#ax2.plot(np.arange(Nx),Trap/np.max(Trap))
	ax2.legend(loc=1);

# Save the figure
ax2.grid(axis='both')
pyplot.show()


temp = '1D_density_Us_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")
'''
pyplot.plot(np.arange(Nx),abs(gauss)**2,'g*')
pyplot.plot(np.arange(Nx),abs(sol_co)**2)
pyplot.grid(axis='both')
pyplot.xlabel('$x$')
pyplot.ylabel('$|\Psi|^2$')
pyplot.show()
'''