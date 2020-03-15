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
N_s = [0,10,15,20,25] # N=0 actually means g = 0
V0 = 0.5*1e-5
Nx = 201
case = ['SC','Harmonic','Isotropic']

# Comparison with Pethick & Smith (eq. 6.20)
g = 0.01
w = np.sqrt(V0*2)
ax = (2*V0)**(-0.25)
print(g/2/np.sqrt(2*np.pi)/ax)
E_PS = np.zeros(len(N_s))
i = 0
for N in N_s:
	E_PS[i] = w/2+g*N/2/np.sqrt(2*np.pi)/ax 
	i += 1

# Figure

fig_OL1D = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

#ax1.set_xlabel('Iterations for conv.')
ax1.set_xlabel('N')
ax1.set_ylabel('$E$')
ax2.set_ylabel('$E$')
ax2.set_xlabel('iterat. for conv.')#,color='r')

E_funct_s = np.array([])

for N in N_s:

	# Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			 	(case[0],case[1],case[2],Nx,J,N,V0)
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data[0]
	E0 = data[1]
	E0_all = data[3]	
	H_KV = data[5]
	print(linalg.eigh(H_KV)[0][0])

	E_funct = data[6]

	eigVal_H_KV,eigVect_H_KV = np.linalg.eigh(H_KV)

	ax1.plot(N,E0_all[-1,0],'ro',label="$\mu$") # mu if lowest energy of H_KV+H_U
	ax1.plot(N,E_funct[-1].real,'go',label="$E[\psi]$")
	ax2.plot(np.arange(len(E0_all[:,0])-1),E0_all[1:,0],'r.',label="$\mu$")
	ax2.plot(np.arange(len(E_funct)),E_funct.real,'g.',label="$E[\psi]$")

	E_funct_s = np.append(E_funct_s,E_funct[-1])

ax1.plot(N_s,E_PS,'-')

print("The angular coefficient of E ~ N is",\
	(E_funct_s[-1]-E_funct_s[-2]).real/(N_s[-1]-N_s[-2]))

# Save the figure

ax1.grid(axis='both')
ax1.legend(loc=5);
ax2.grid(axis='both')
ax2.legend(loc=5);
pyplot.show()

temp = '1D_Energies_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")