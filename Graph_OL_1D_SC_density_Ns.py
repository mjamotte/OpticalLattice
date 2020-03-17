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

N_s = [25]#0,9,10,11,24,25,26,49,50,51]

## Figure

fig_OL1D = pyplot.figure(figsize=(8,8))
ax2 = pyplot.subplot(111)
ax2.set_xlabel('$r_{center}/\sigma$')
ax2.set_ylabel('$|\psi_0|^2$')#,color='r')

for N in N_s:

	## Parameters
	args_syst = {
	'J' : 1,
	'N' : N, # N=0 actually means g = 0 
	'V' : 0.5*1e-3,
	'Nx' : 51,
	'U' : 0.01,
	'Method' : 'SC',
	'Trap' : 'Harmonic',
	'Symm' : 'Isotropic',
	}

	J = args_syst['J']
	Nx = args_syst['Nx']
	V = args_syst['V']

	## Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%1i_V_%.4f' %\
			 	(args_syst['Method'],args_syst['Trap'],\
			 	args_syst['Symm'],\
			 	args_syst['Nx'],args_syst['J'],args_syst['N'],\
			 	args_syst['V'])
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)

	args_init = data[1]
	mu_all = data[2]
	psi0 = data[3]
	E_funct_SC = data[4]

	n0 = np.abs(psi0)**2

	## Analytical gaussien 

	Trap = Fun_OL.trap_1D(args_syst)
	m = 1/(2*J)
	w0 = np.sqrt(V/m)
	sigma = np.sqrt(1/(m*w0))
	x0 = (Nx-1)/2
	positions = np.arange(Nx)
	Gauss = Fun_OL.gauss(positions,x0,sigma) # Lowest band of HKV (analytic)

	centered = 'False'
	if centered=='True':
		half_len = int(len(positions)/2)
		center = positions[half_len]
		half_pos = (positions-center)[half_len:]
		ax2.plot(half_pos/sigma,Gauss[half_len:]**2/np.sum(Gauss**2),'-g')
		ax2.plot(half_pos/sigma,n0[half_len:].real,'-r',label="$N = {}$".format(N))

	if centered=='False':
		ax2.plot(positions,Gauss**2/np.sum(Gauss**2),'-g')
		ax2.plot(positions,n0,'-',label="$N = {}$".format(N))

	#ax2.plot(np.arange(Nx),Trap/np.max(Trap))
	ax2.legend(loc=1);

## Save the figure
ax2.grid(axis='both')
pyplot.show()

temp = '1D_density_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(args_syst['Method'],args_syst['Trap'],\
		args_syst['Symm'],args_syst['Nx'],args_syst['J'],\
		args_syst['V'])
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")

'''
J = 1
N_s = [0,10,25,50] # N=0 actually means g = 0
V0 = 0.5*1e-3
Nx = 51
case = ['SC','Harmonic','Isotropic']

# Figure

fig_OL1D = pyplot.figure(figsize=(8,8))
ax2 = pyplot.subplot(111)
ax2.set_xlabel('$r_{center}/\sigma$')
ax2.set_ylabel('$|\psi_0|^2$')#,color='r')

for N in N_s:

	# Import datas to plot
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			 	(case[0],case[1],case[2],Nx,J,N,V0)
	data = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data[0]
	E0 = data[1]
	n0 = data[2]
	E0_all = data[3]
	n0_all = data[4]
	H_KV = data[5]
	E_funct = data[6]

	# Analytical gaussien 

	Trap = Fun_OL.trap_1D(system,case)
	m = 1/(2*abs(J))
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
		ax2.plot(half_pos/sigma,n0[half_len:].real,'-r',label="$N = {}$".format(U))

	if centered=='False':
		ax2.plot(positions,Gauss**2/np.sum(Gauss**2),'-g')
		ax2.plot(positions,dens,'.b')
		ax2.plot(positions,n0.real,'-',label="$N = {}$".format(N))

	#ax2.plot(np.arange(Nx),Trap/np.max(Trap))
	ax2.legend(loc=1);

# Save the figure
ax2.grid(axis='both')
pyplot.show()


temp = '1D_density_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")'''