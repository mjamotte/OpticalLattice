import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import cmath
import time
from mpl_toolkits import mplot3d
from scipy.integrate import ode
import pylab

#import nbimporter
import Fun_OL
import debugger as bug

##################TEST OL 1D ##########################

debug = True

params_fig_density = { 
	'xlabel':'x',
	'ylabel':'$|\Psi_0|^2$',
	'pause' : 0.01
}

params_fig_mu = {
	'pause' : 0.01,
	'xlabel' : 'Iterations',
	'ylabel' : '$\mu$'
}

args_plot = {
	'params_fig' : params_fig_density
	#'params_fig' : params_fig_mu
}

############### CODE ###################
N_s = [25,26,49,50,51]
k = 0
for N in N_s:

	start = time.time()

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

	args_syst.update({'sites_dic' : Fun_OL.lattice_1D(args_syst)})

	## Kinetic + Trap part of Hamiltonian
	H_KV = Fun_OL.H_1D(args_syst)

	if k<=1:
		args_init = {
		'H_KV' : H_KV,
		'dt' : 5*1e-3,
		'err_IT' : 1e-9
		}

	else:
		args_init.update({'psi_int' : psi0})
	
	mu_all,psi0,E_funct_IT = Fun_OL.calc_psi0(args_syst,args_init)

	data = [args_syst,args_init,mu_all,psi0,E_funct_IT]
	dataID = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			(args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
			args_syst['Nx'],args_syst['J'],N,args_syst['V'])
	np.save('Datas/'+dataID+'.npy',data)

	print("Execution time:",time.time()-start,"secondes")
	k += 1