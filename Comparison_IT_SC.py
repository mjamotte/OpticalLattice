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

#### COMPARISON OF ENERGIES AND DENSITIES FOR IT AND SC ######

N_s = [0,9,10,11,24,25,26,49,50,51] # N=0 actually means U = 0 

## Figure

fig_OL1D = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

ax1.set_xlabel('N')
#ax1.set_ylabel('$E/N,\mu$')
ax2.set_ylabel('$n_I,n_S$')
ax2.set_xlabel('$x$')#,color='r')

Efun_SC = np.array([]) # to compute mu = dE_SC/dN
Efun_IT = np.array([]) # to compute mu = dE_IT/dN
k = 0 # to print the legend at the right time

for N in N_s:

	args_syst = {
	'J' : 1,
	'N' : N, # N=0 actually means g = 0 
	'V' : 0.5*1e-3,
	'Nx' : 51,
	'U' : 0.01,
	'Method' : 'IT',
	'Trap' : 'Harmonic',
	'Symm' : 'Isotropic',
	}

## Import relevant datas

	## Import datas for IT
	
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%1i_V_%.4f' %\
				 (args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
			args_syst['Nx'],args_syst['J'],N,args_syst['V'])
	data_IT = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	
	args_init = data_IT[1]
	mu_all = data_IT[2]
	psi0_IT = data_IT[3]
	n0_IT = np.abs(psi0_IT)**2
	E_funct_IT = data_IT[4]

	## Import datas for SC

	args_syst.update({'Method' : 'SC'})
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%1i_V_%.4f' %\
				 (args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
			args_syst['Nx'],args_syst['J'],N,args_syst['V'])
	data_SC = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	
	args_init = data_SC[1]
	E0_all = data_SC[2]
	psi0_SC = data_SC[3]
	n0_SC = np.abs(psi0_SC)**2
	E_funct_SC = data_SC[4]

	Efun_IT = np.append(Efun_IT,E_funct_IT)
	Efun_SC = np.append(Efun_SC,E_funct_SC)

	J = args_syst['J']
	Nx = args_syst['Nx']
	V = args_syst['V']

## Comparison energies' graphs (and chemical potential)
	j = int(len(mu_all[-1])/2)
	if k<len(N_s)-1:
		k += 1
		ax1.plot(N,E_funct_IT/N,'ks',fillstyle='none')
		ax1.plot(N,E_funct_SC/N,'y*')
		ax1.plot(N,mu_all[-1,j],'gs',fillstyle='none')
		ax1.plot(N,E0_all[-1,0],'r*')
	else:
		ax1.plot(N,E_funct_IT/N,'ks',fillstyle='none',label="$E_I[\psi]/N$")
		ax1.plot(N,E_funct_SC/N,'y*',label="$E_S[\psi]/N$")
		ax1.plot(N,mu_all[-1,j],'gs',fillstyle='none',label="$\mu_I$")
		ax1.plot(N,E0_all[-1,0],'r*',label="$E^0_S$")

## Comparison densities' graphs

	if N==0 or N==10 or N==25 or N==50:	# not every N for clarity on plot
		## Analytical Gaussian 
		m = 1/(2*abs(J))
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
			ax2.plot(half_pos/sigma,dens[half_len:],'.b')
			ax2.plot(half_pos/sigma,n0[half_len:].real,'-r',label="$N = {}$".format(U))

		if centered=='False':
			ax2.plot(positions/sigma,Gauss**2/np.sum(Gauss**2),'-g')
			ax2.plot(positions/sigma,n0_IT,'-',label="$IT, N = {}$".format(N))
			ax2.plot(positions/sigma,n0_SC,'*',label="$SC, N = {}$".format(N))

## Comparison of values

	## Compute mu = dE/dN
dN = 1
dEfun_SCdN = Fun_OL.dEdN_O2(Efun_SC[1:],dN)
dEfun_ITdN = Fun_OL.dEdN_O2(Efun_IT[1:],dN)
ax1.plot([10,25,50],dEfun_SCdN,'ob',fillstyle='none',label="$dE_S/dN$")
ax1.plot([10,25,50],dEfun_ITdN,'oc',fillstyle='none',label="$dE_I/dN$")

## Grid, legend and save the figure

pyplot.suptitle('Comparison IT and SC for %s, %s,\
	 Nx = %.1i, J = %.2f, V = %.4f' % \
	 (args_syst['Trap'],args_syst['Symm'],\
	args_syst['Nx'],args_syst['J'],args_syst['V']))

ax1.legend(loc=2);
ax2.legend(loc=1);
ax1.grid(axis='both')
ax2.grid(axis='both')
pyplot.show()

temp = '1D_comp_ITSC_Ns_%s_%s_Nx_%.1i_J_%.2f_V_%.4f' %\
		(args_syst['Trap'],args_syst['Symm'],\
		 args_syst['Nx'],args_syst['J'],args_syst['V'])
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")