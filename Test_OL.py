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

import nbimporter
import Fun_OL

##################TEST OL ##########################

J = 1
U = 0
V0 = 10**-5
Nx = 1000
Ny = 2
a1 = np.array([1,0])
a2 = np.array([0,1])

sites_dic = Fun_OL.square_lattice(Nx,Ny,a1,a2)

system = [sites_dic,J,U,V0,Nx,Ny]
case = ['SC','Harmonic','Isotropic']

start = time.time()

H_KV = Fun_OL.construct_H_KV(system,case)

#pyplot.pcolormesh(H_KV.real)
#pyplot.colorbar()

E0,n0,E0_all = Fun_OL.density(case,system,H_KV)

print("Execution time:",time.time()-start,"secondes")

data = [system,E0,n0,E0_all,H_KV]
dataID = '%s_%s_%s_Nx_%.1iNy_%.1i_J_%.2f_U_%.4f_V_%.4f' %\
		 (case[0],case[1],case[2],Nx,Ny,system[1],system[2],system[3])
np.save('Datas/'+dataID+'.npy',data)