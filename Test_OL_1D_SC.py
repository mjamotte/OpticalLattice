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

#import nbimporter
import Fun_OL

##################TEST OL 1D ##########################

J = 1
N_s = [100]#[0,9,10,11,24,25,26,49,50,51] # N=0 actually means g=0 
V = 0#0.5*1e-3
Nx = 51

sites_dic = Fun_OL.lattice_1D(Nx)
eigVal = 1j*np.zeros(Nx)
eigVect = 1j*np.zeros((Nx,Nx))

case = ['SC','Harmonic','Isotropic']

for N in N_s:
	start = time.time()

	system = [sites_dic,J,N,V,Nx]
	H_KV = Fun_OL.H_1D(system,case)

	E0,n0,E0_all,n0_all,E_funct_SC = Fun_OL.density(case,system,H_KV)

	print("Execution time:",time.time()-start,"secondes")

	data = [system,E0,n0,E0_all,n0_all,H_KV,E_funct_SC]
	dataID = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.1i_V_%.4f' %\
			(case[0],case[1],case[2],Nx,system[1],system[2],system[3])
	np.save('Datas/'+dataID+'.npy',data)

