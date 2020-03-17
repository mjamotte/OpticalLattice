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

J = 1
V = 0.5*1e-3
N_s = [0,9,10,11,24,25,26,49,50,51]
Nx = 51

sites_dic = Fun_OL.lattice_1D(Nx)
case = ['IT','Harmonic','Isotropic']

for N in N_s:
	start = time.time()

	system = [sites_dic,J,N,V,Nx]

	## lowest state without int. = gaussian as initial state

	H_KV = Fun_OL.H_1D(system,case)

	mu_all,n0,E_funct_IT = Fun_OL.density(case,system,H_KV)

	data = [system,mu_all,n0,H_KV,E_funct_IT]
	dataID = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			(case[0],case[1],case[2],Nx,system[1],system[2],system[3])
	np.save('Datas/'+dataID+'.npy',data)

	print("Execution time:",time.time()-start,"secondes")