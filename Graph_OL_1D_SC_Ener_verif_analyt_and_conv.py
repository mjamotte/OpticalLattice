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
U_s = [0,0.1,0.15,0.2,0.25]#np.array([0,0.1,0.5,1,5,10])
V0 = 0.5*1e-5
Nx = 201
case = ['SC','Harmonic','Isotropic']

# Figure

fig_OL1D = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

#ax1.set_xlabel('Iterations for conv.')
ax1.set_xlabel('U')
ax1.set_ylabel('$E$')
ax2.set_ylabel('$E$')
ax2.set_xlabel('iterat. for conv.')#,color='r')

for U in U_s:

	# Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_U_%.4f_V_%.4f' %\
			 	(case[0],case[1],case[2],Nx,J,U,V0)
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data[0]
	E0 = data[1]
	n0 = data[2]
	E0_all = data[3]
	n0_all = data[4]
	H_KV = data[5]
	E_funct = data[6]

	eigVal_H_KV,eigVect_H_KV = np.linalg.eigh(H_KV)

	# Analytical gaussien 

	Trap = Fun_OL.trap_1D(system,case)
	m = 1/(2*J)
	w0 = np.sqrt(V0/m)
	sigma = np.sqrt(1/(m*w0))
	x0 = (Nx-1)/2
	positions = np.arange(Nx)

	ax1.plot(U,E0_all[-1,0],'ro',label="$\mu$") # mu if lowest energy of H_KV+H_U
	ax1.plot(U,E_funct[-1].real,'go',label="$E[\psi]$")
	ax2.plot(np.arange(len(E0_all[:,0])-1),E0_all[1:,0],'r.',label="$\mu$")
	ax2.plot(np.arange(len(E_funct)),E_funct.real,'g.',label="$E[\psi]$")


# Save the figure

ax1.grid(axis='both')
ax1.legend(loc=2);
ax2.grid(axis='both')
ax2.legend(loc=2);
pyplot.show()

temp = '1D_Energies_Us_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")