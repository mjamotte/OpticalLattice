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
'''
a = []#np.transpose([np.array([])])
print(a)
b = [np.array([4,5,6])]
c = np.append(a,np.transpose(b),axis=1)
print(c[:,:])
exit()
'''

J = 1
U_s = [0]#,0.1,0.15,0.2,0.25]#np.array([0,0.1,0.5,1,5,10])
V = 0.5*1e-5
Nx = 201
Ny = 0

sites_dic = Fun_OL.lattice_1D(Nx)
eigVal = 1j*np.zeros(Nx)
eigVect = 1j*np.zeros((Nx,Nx))

case = ['IT','Harmonic','Isotropic']

start = time.time()

for U in U_s:
	system = [sites_dic,J,U,V,Nx,Ny]
	H_KV = Fun_OL.H_1D(system,case)
	psi0 = linalg.eigh(H_KV)[1][:,0]

	system = [sites_dic,J,U,V,Nx,Ny,psi0]
	E0_all,n0_all = Fun_OL.density(case,system,H_KV)

	print("Execution time:",time.time()-start,"secondes")

	data = [system,E0_all,n0_all,H_KV]
	dataID = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_U_%.4f_V_%.4f' %\
			(case[0],case[1],case[2],Nx,system[1],system[2],system[3])
	np.save('Datas/'+dataID+'.npy',data)