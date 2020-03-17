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

J = 1
N_s = [0,9,10,11,24,25,26,49,50,51] # N=0 actually means g = 0 
V0 = 0.5*1e-3
Nx = 51

## Figure

fig_OL1D = pyplot.figure(figsize=(20,8))
ax1 = pyplot.subplot(121)
ax2 = pyplot.subplot(122)

ax1.set_xlabel('N')
ax1.set_ylabel('$E,\mu$')
ax2.set_ylabel('$n_I,n_S$')
ax2.set_xlabel('$x$')#,color='r')

Efun_SC = np.array([]) # to compute mu = dE/dN

for N in N_s:

## Import relevant datas

	## Import datas for IT
	case = ['IT','Harmonic','Isotropic']
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%1i_V_%.4f' %\
				 (case[0],case[1],case[2],Nx,J,N,V0)
	data_IT = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	system = data_IT[0]
	mu_all = data_IT[1]
	n0_IT = data_IT[2]
	E_funct_IT = data_IT[4]

	## Import datas for SC
	case = ['SC','Harmonic','Isotropic']
	temporary = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%1i_V_%.4f' %\
				 (case[0],case[1],case[2],Nx,J,N,V0)
	data_SC = np.load('Datas/'+temporary+'.npy',allow_pickle=True)
	n0_SC = data_SC[2]
	E0_all = data_SC[3]
	H_KV = data_SC[5]
	E_funct_SC = data_SC[6]
	Efun_SC = np.append(Efun_SC,E_funct_SC[-1])#*N*Nx)

## Comparison energies' graphs (and chemical potential)

	ax1.plot(N,E_funct_IT,'ks',label="$E_I[\psi], N = {}$".format(N))
	ax1.plot(N,E_funct_SC[-1],'y*',label="$E_S[\psi], N = {}$".format(N))
	ax1.plot(N,mu_all[0,0],'gs',label="$\mu_I, N = {}$".format(N))
	ax1.plot(N,E0_all[-1,0],'r*',label="$E^0_S, N = {}$".format(N))

## Comparison densities' graphs

	if N==0 or N==10 or N==25 or N==50:	# not every N for clarity on plot
		## Analytical Gaussian 
		m = 1/(2*abs(J))
		w0 = np.sqrt(V0/m)
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
			ax2.plot(positions/sigma,n0_IT.real,'-',label="$IT, N = {}$".format(N))
			ax2.plot(positions/sigma,n0_SC.real,'*',label="$SC, N = {}$".format(N))

## Comparison of values

	## COmpute mu = dE/dN
dN = 1
dEfun_SCdN = Fun_OL.dEdN_O2(Efun_SC[1:],dN)

ax1.plot([10,25,50],dEfun_SCdN,'ob')

## Grid, legend and save the figure

#ax1.legend(loc=3);
ax2.legend(loc=1);
ax1.grid(axis='both')
ax2.grid(axis='both')
pyplot.show()

temp = '1D_comp_ITSC_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.4f' %\
		(case[0],case[1],case[2],Nx,J,V0)
fig_OL1D.savefig("Figures_OL/fig_"+temp+".pdf")