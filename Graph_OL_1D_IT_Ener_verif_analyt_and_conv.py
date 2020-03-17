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
'''
This code is used to compare to analytical relations between the energies, 
the number of particles and the convergence of the self-consistent method 
in minimizing E-mu*N 
'''


J = 1
N_s = [0,10,25,50] # N=0 actually means g = 0
V0 = 0.5*1e-3
Nx = 51
case = ['IT','Harmonic','Isotropic']

# Figure

fig_OL1D = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

#ax1.set_xlabel('Iterations for conv.')
ax1.set_xlabel('N')
ax1.set_ylabel('$E$')
ax2.set_ylabel('$mu$')
ax2.set_xlabel('iterat. for conv.')#,color='r')

E_funct_IT_s = np.array([])

for N in N_s:

	# Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			 	(case[0],case[1],case[2],Nx,J,N,V0)
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data[0]
	mu_all = data[1]	
	H_KV = data[3]
	E_funct_IT = data[4]

	ax1.plot(N,E_funct_IT.real,'go',label="$E[\psi]$")
	#ax2.plot(np.arange(len(E_funct_IT)),E_funct_IT.real,'g.',label="$E[\psi]$")
	for x in range(len(mu_all[0])):
		pyplot.plot(np.arange(len(mu_all)),mu_all[:,x].real) # mu.imag is zero

	E_funct_IT_s = np.append(E_funct_IT_s,E_funct_IT)

# Save the figure

ax1.grid(axis='both')
ax1.legend(loc=5);
ax2.grid(axis='both')
ax2.legend(loc=5);
pyplot.show()

temp = '1D_Energies_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")
'''
for x in range(len(mu_all[0])):
	pyplot.plot(np.arange(len(mu_all)),mu_all[:,x].real) # mu.imag is zero

pyplot.xlabel('Time')
pyplot.ylabel('$\mu$')

pyplot.show()
'''