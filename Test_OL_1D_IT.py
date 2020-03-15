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

############## DEBUGGER PART ##############

debug = False

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
}

############### CODE ###################
J = 1
N = 25#[0,10,15,20,25] # N=0 actually means g = 0 
V = 0.5*1e-4
Nx = 201

sites_dic = Fun_OL.lattice_1D(Nx)
system = [sites_dic,J,N,V,Nx]
case = ['IT','Harmonic','Isotropic']
H_KV = Fun_OL.H_1D(system,case)

## lowest state without int. = gaussian
gauss = (linalg.eigh(H_KV)[1][:,0])
psi_old = np.concatenate((np.real(gauss), np.imag(gauss)))

## parameters for set_integrator and GP
tol = 1e-9 # tolerance
nsteps = np.iinfo(np.int32).max
solver = ode(Fun_OL.GP)
solver.set_f_params(system,H_KV) # parameters needed in GP_t_real
solver.set_integrator('dop853', atol=tol, rtol=tol, nsteps=nsteps)

## Evolution
t = 0
dt = 0.05
half_len = int(len(gauss)/2)
err = 1e-9
counterIT = 0
dim = len(psi_old)
positions = np.arange(Nx)

mu_old = 0
flag = 0

while True:
	## time evolution
	solver.set_initial_value(psi_old, t)
	solver.integrate(t+dt)
	t = t+dt
	psi_new = solver.y

	# Compute mu
	psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]
	psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):] # not renorm. yets
	mu_new = -np.log(psi_new_co/psi_old_co)/dt
	if flag==0:
		#a = np.array([[1,2,3],[4,5,6]])
		
		mu_all = np.array([mu_new])
		flag = 1
	else:
		#a = np.append(a,np.array([[7,8,9]]),axis=0)
		mu_all = np.append(mu_all,np.array([mu_new]),axis=0)

	## renormalize
	psi_new = psi_new/np.sqrt(np.sum(abs(psi_new)**2))
	psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):]

	#print('Real Part =', psi_new_co[half_len].real,\
	 #'Imag Part =', psi_new_co[half_len].imag, 'module =', abs(psi_new_co[half_len])**2)
	#print('Real Part =', psi_co[half_len].real,\
	 #'Imag Part =', psi_co[half_len].imag, 'module =', abs(psi_co[half_len])**2)

	if debug==True:
		pyplot.figure(1)
		bug.debug_plot_vector(abs(psi_new_co)**2,args_plot)

		pyplot.figure(2)
		bug.debug_plot_vector(mu_all,args_plot)
		bug.debug_plot_vector(mu2_all,args_plot)


	err1 = np.sqrt(np.sum((abs(abs(psi_new_co)**2-abs(psi_old_co)**2)**2)))

	#if abs(mu_old-mu_new)/abs(mu_new)<err:
	if err1<err:
		break

	psi_old = psi_new

	mu_old = mu_new
	counterIT += 1

if solver.successful():
	sol = solver.y
	sol = sol/np.sqrt(sum(abs(sol)**2))
	sol_re = sol[:int(dim/2)]
	sol_im = sol[int(dim/2):]
	sol_co = sol_re + 1j*sol_im

print('The number of iterations for IT =', counterIT)


pyplot.plot(np.arange(Nx),abs(gauss)**2,'g*')
pyplot.plot(np.arange(Nx),abs(sol_co)**2)
pyplot.grid(axis='both')
pyplot.xlabel('$x$')
pyplot.ylabel('$|\Psi|^2$')
pyplot.show()

for x in range(len(mu_all[0])):
	pyplot.plot(np.arange(len(mu_all)),mu_all[:,x].real) # mu.imag is zero

pyplot.xlabel('Time')
pyplot.ylabel('$\mu$')

pyplot.show()


'''
J = 1
N_s = [0]#,0.1,0.15,0.2,0.25]#np.array([0,0.1,0.5,1,5,10])
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

	system = [sites_dic,J,N,V,Nx,Ny,psi0]
	E0_all,n0_all = Fun_OL.density(case,system,H_KV)

	print("Execution time:",time.time()-start,"secondes")

	data = [system,E0_all,n0_all,H_KV]
	dataID = '1D_%s_%s_%s_Nx_%.1i_N_%.1i_V_%.4f' %\
			(case[0],case[1],case[2],Nx,system[2],system[3])
	np.save('Datas/'+dataID+'.npy',data)
	'''