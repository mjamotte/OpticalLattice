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
from numba import jit
import debugger as bug
############# FUNCTIONS OPTICAL LATTICES #############

'''

^ y	  |	 |	|  |  |
|	  A	 B--A  B--A
|	  |  |  |  |  |
|	  B--A  B--A  B
|	  |  |  |  |  |
|	  A  B--A  B--A
|	(0,0)
|-------------------> x

'''

def square_lattice(Nx,Ny,a1,a2):

####################################################################################
#																				   #	
#	Constructs a square lattice with Bravais vectors a1 and a2, and composed of Nx # 
# 	sites in the x direction and Ny sites in the y direction.					   #
#																				   #	
####################################################################################
	 
	sites_dic = {}
	n = 0
	for j in range(Ny):
		for i in range(Nx): # fill line by line, from left to right
			sites_dic[n] = np.array([i,j])
			n += 1
			
	return sites_dic
	

def trap(system,case):

####################################################################################
#																				   #	
#	Trap-potential of amplitude V0 at position (i,j) that is a lattice site.	   #
#	One needs to calculate x and y from (i,j) depending on the lattice.			   #
#																				   #
#	Inputs:	- case: indicates the the approximation used: the kind of 		       #
#					Trap-potential												   #
#					used, the boundary	   										   #
#					conditions, the method to solve GP; 						   #
#					case = [resolution method, potential, boundary conditions]	   #
#			- i : integer indicating the lattice site which we calculate the       #
#			  	  potential at.											   	   	   #
#			- system: list of parameters = [sites_dic,U,V0,Nx,Ny,				   #
#											mu,J,kxa_list,kya_list]				   #
#			   																	   #	
####################################################################################
		
	sites_dic = system[0]
	V0 = system[2]
	Nx = system[3]
	Ny = system[4]
	L = len(sites_dic)

	Lx = (Nx-1)*3/2
	Ly = Ny*np.sqrt(3)/2

	out = np.zeros(L)

	if case[1]=='Linear': #linear potential

		for i in range(L):

			x = sites_dic[i][0]*3/2
			y = sites_dic[i][1]*np.sqrt(3)/2

	if case[1]=='Harmonic':

		for i in range(L):

			x = sites_dic[i][0]*3/2 # a=1
			#y = sites_dic[i][1]*np.sqrt(3)/2
			#out[i] = 1/2*V0*np.cos(2/3*x*np.pi) # Periodic Harmonic trap along x
			out[i] = 1/2*V0*x**2

	return out


def honey_hopping(n1,n2,system,delta_link):

##########################################################
#
# 	Returns the hopping parameters from lattice site n1 to n2. This
#	allows to deform the honeycomb lattice where the BEC is trapped.		#
#	
#
	#sites_dic = system[0]
	#xA = sites_dic[n1][0]*3/2
	#yA = sites_dic[n1][1]*np.sqrt(3)/2
	#Nx = system[1]
	#Ny = system[2]
	#Lx = 3/2*(Nx-1)
	#Ly = Ny*np.sqrt(3)/2
	J = system[5]
	#gamma2 = system[5]
	#gamma3 = system[6]

	if delta_link=='delta_1':
		hop = -J

	elif delta_link=='delta_2':
		hop = -J

	elif delta_link=='delta_3':
		hop = -J

	return hop


def calc_psi0(args_syst,args_init):

	if args_syst['Method']=='SC': #SC = self-consistent method
		E0_all,psi0_all,E_funct_SC = solveGP_SC(args_syst,args_init)
		return E0_all,psi0_all,E_funct_SC

	elif args_syst['Method']=='IT': #Imaginary time
		mu_all,psi0,E_funct_IT = solve_GP_IT(args_syst,args_init)
		return mu_all,psi0,E_funct_IT


def solveGP_SC(args_syst,args_init):

	N = args_syst['N']
	H_KV = args_init['H_KV']
	err = args_init['err_SC'] # condition to stop convergence
	E_funct = np.array([])

	
	E0_all = [np.zeros(len(H_KV))] # to observe convergence of the lowest energy
	psi0_all = [np.array([])]
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap	

	#E0_temp1,eigVecs = sc.sparse.linalg.eigsh(H_KV,which='SA',k=1)
	E0_old,eigVecs = linalg.eigh(H_KV)
	psi_old = np.matrix.transpose(eigVecs[:,0])
	#psi0_temp = impose_sym(psi0_temp)

	E0_all = [E0_old]
	#psi0_all = np.append(psi0_all,psi0_temp)

	counterSC = 0
	lam = 1/(10*N+1) # to avoid oscillations in energy that slow down the code
	#flag = 0

	while True:

		#H_U = sc.sparse.coo_matrix(0.5*U*np.diag(np.abs(psi0_temp)**2))
		#E0_temp2,eigVecs = sc.sparse.linalg.eigsh(H_KV+H_U,which='SA',k=1)
		H_U = H_int(psi_old,args_syst)
		E0_new,eigVecs = linalg.eigh(H_KV+H_U)
		psi_new = np.matrix.transpose(eigVecs[:,0])

		psi_lam = np.sqrt((1-lam)*psi_old**2 + lam*psi_new**2)

		#if len(E_funct)>2:
		#	if E_funct[-1]>E_funct[-2] and flag==0:
		#		flag = 1
		#		print('Energy functional is not decreasing')

		E0_all = np.append(E0_all,[E0_new],axis=0)
		#psi_all = np.append(psi0_all,psi0_lam,axis=0)

		#if abs((E0_temp1[0]-E0_temp2[0])/E0_temp1[0])<err:
		if np.sum(abs(np.abs(psi_old)**2-np.abs(psi_lam)**2))<err:
			break

		psi_old = psi_lam
		E0_old = E0_new
		counterSC += 1			

	print('Number of iterations of self-consistent method =',counterSC)
	E_funct = energy_functional(psi_lam,args_syst)

	return E0_all,psi_lam,E_funct #,psi_all


def H_int(psi,args_syst):
	U = args_syst['U']
	N = args_syst['N']

	return U*N*np.diag(np.abs(psi)**2)

def energy_functional(psi,args_syst):

	J = args_syst['J']
	N = args_syst['N']
	V = args_syst['V']
	U = args_syst['U']
	Nx = args_syst['Nx']

	E_U = U*N/2*np.sum(np.abs(psi)**4) # -U/2 let it for small systems
	E_trap = np.sum(trap_1D(args_syst)*abs(psi)**2)
	
	positions = np.arange(Nx-1) # -1 because of the k+1 below
	E_kin = 0
	for k in positions:
		E_kin += -J*np.conj(psi[k])*psi[k+1]-J*np.conj(psi[k+1])*psi[k]

	return (E_U+E_trap+E_kin)*N


def GP(t,psi_old,args_syst,args_init):

	H_KV = args_init['H_KV']
	N = args_syst['N']
	dim = len(psi_old)
	psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]

	# Hopping part + trap part of the GP
	y1 = H_KV.dot(psi_old_co)
	# Interacting part of the GP
	y2 = H_int(psi_old_co,args_syst).dot(psi_old_co)

	y = y1 + y2
	# -d_tau psi(tau) = H psi(tau)
	y = -y
	return np.concatenate((np.real(y),np.imag(y)))


def solve_GP_IT(args_syst,args_init):

	## Initialisation 
	if 'psi0' in args_init:
		psi_old = args_init['psi0'] 
		psi_old = np.concatenate((np.real(psi_old), np.imag(psi_old)))

	else:
		H_KV = args_init['H_KV']
		gauss = linalg.eigh(H_KV)[1][:,0]
		psi_old = np.concatenate((np.real(gauss), np.imag(gauss)))

	## parameters for set_integrator and GP
	tol = 1e-9 # tolerance
	nsteps = np.iinfo(np.int32).max
	solver = ode(GP)
	solver.set_f_params(args_syst,args_init) # parameters needed in GP_t_real
	solver.set_integrator('dop853', atol=tol, rtol=tol, nsteps=nsteps)

	## Evolution
	t = 0
	dt = args_init['dt']
	err = args_init['err_IT']
	counterIT = 0
	dim = len(psi_old)
	#mu_old = 0
	flag = 0

	while True:
		## time evolution
		solver.set_initial_value(psi_old, t)
		solver.integrate(t+dt)
		t = t+dt
		psi_new = solver.y

		## Compute mu
		psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]
		psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):] # not renorm. yet
		mu_new = -np.log(psi_new_co/psi_old_co)/dt

		if flag==0:
			mu_all = np.array([mu_new])
			flag = 1
		else:
			mu_all = np.append(mu_all,np.array([mu_new]),axis=0)

		## renormalize
		psi_new = psi_new/np.sqrt(np.sum(abs(psi_new)**2))
		psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):]

		# if debug==True:
		# 	pyplot.figure(1)
		# 	bug.debug_plot_vector(abs(psi_new_co)**2,args_plot)

		# 	pyplot.figure(2)
		# 	bug.debug_plot_vector(mu_all,args_plot)

		err1 = np.sqrt(np.sum((abs(abs(psi_new_co)**2-abs(psi_old_co)**2)**2)))

		#if abs(mu_old-mu_new)/abs(mu_new)<err:
		if err1<err:
			break

		psi_old = psi_new

		#mu_old = mu_new
		counterIT += 1

	if solver.successful():
		sol = solver.y
		sol = sol/np.sqrt(sum(abs(sol)**2))
		sol_re = sol[:int(dim/2)]
		sol_im = sol[int(dim/2):]
		psi0 = sol_re + 1j*sol_im

		E_funct_IT = energy_functional(psi0,args_syst)

	print('The number of iterations for IT =', counterIT)

	return mu_all,psi0,E_funct_IT


def impose_sym(vector): 
# imposes symmetry from the middle of the vector
	Len = len(vector)

	for i in range(int(Len/2)):
		vector[i] = (vector[i]+vector[Len-i-1])*0.5
		vector[Len-i-1] = vector[i]

	norm = np.sqrt(np.sum(np.abs(vector)**2))
	vector = vector/norm
	return vector

def dEdN_O2(E,dN):

	dEdN = np.array([])
	for i in range(int(len(E)/3)): # 3 = "order of approx"+1
		dEdN = np.append(dEdN,(E[i*3+2]-E[i*3])/(2*dN))
	return dEdN


############# 1D OPTICAL LATTICE #############

''' 
	The goal of coding this is the understanding of the physics 
	of the Hamiltonian presented in the Lühmann's paper to 
	aquire feelings with correlated hoppings. 
'''

def lattice_1D(args_syst):
	Nx = args_syst['Nx']

	sites_dic = {}
	n = 0
	for i in range(Nx): # fill line by line, from left to right
		sites_dic[n] = np.array([n])
		n += 1
			
	return sites_dic

	return 	

def trap_1D(args_syst):

	V0 = args_syst['V']
	Nx = args_syst['Nx']

	V = np.zeros(Nx)

	for x in range(Nx):
		V[x] = 1/2*V0*(x-(Nx-1)/2)**2

	return V


def H_1D(args_syst):

	J = args_syst['J']
	Nx = args_syst['Nx']

	H = 1j*np.zeros((Nx,Nx))

	if args_syst['Symm']=='Isotropic':

		if args_syst['Trap']=='Harmonic':
			H = np.diag(trap_1D(args_syst))

		for i in range(Nx-1):
			H[i,i+1] = -J
			H[i+1,i] = np.conj(H[i,i+1])

	return H

def gauss(xs,x0,sigma): # analytical

	Gauss = np.zeros(len(xs))
	i = 0
	for x in xs:
		Gauss[i] = 1/np.sqrt(sigma**2*2*np.pi)*np.exp(-(x-x0)**2/(2*sigma**2))
		i += 1

	return Gauss

def vector2matrix(vector,Nx,Ny):

	# Shape a vector into a Nx x Ny matrix
	L = len(vector)
	if L==Nx*Ny:
		mat = np.zeros((Ny,Nx))

		for i in range(Ny):
			for j in range(Nx):
				mat[i,j] = vector[i*Nx+j]

	else:
		print('Sizes do not match')
		mat = 0

	return mat

def corr_hopping(n1,n2,density,system,delta_link):

	t = system[1]

	if delta_link=='delta_1':
		hop = -t 

	if delta_link=='delta_2':
		hop = -t

	if delta_link=='delta_3':
		hop = -t

	return hop

		
def construct_H_bc(system,case,n):

	Nx = system[4]
	Ny = system[5]
	t = system[1]

	H_bc = np.zeros((Nx*Ny,Nx*Ny))

	for n in range(Nx*Ny):

		if n%(2*Nx)<Nx: # line beginning with A site

			if n%2!=0 and (n+1)%Nx!=0: # not on the right boarder
				# n = B site to n+1 = A site
				H_bc[n+1,n] = corr_hopping(n+1,n,system,'delta_1') #link t1
				H_bc[n,n+1] = np.conj(H_bc[n+1,n])

			if n<Nx*(Ny-1) and n%2==0:
				# n = A site to n+Nx = B site
				H_bc[n,n+Nx] = corr_hopping(n,n+Nx,system,'delta_2') #link t2
				H_bc[n+Nx,n] = np.conj(H_bc[n,n+Nx])

				if n<Nx*(Ny-1)-1:
					# n+1 = B site to n+1+Nx = A site
					H_bc[n+1+Nx,n+1] = corr_hopping(n+1+Nx,n+1,system,'delta_3') # link t3
					H_bc[n+1,n+1+Nx] = np.conj(H_bc[n+1+Nx,n+1])

			elif n%(2*Nx)>=Nx: # line beginning with B site

				if Nx%2!=0: # line begins and ends with same site-type

					if n%2!=0 and (n+1)%Nx!=0:
						# n = B site to n+1 = A site
						H_bc[n+1,n] = corr_hopping(n+1,n,system,'delta_1')
						H_bc[n,n+1] = np.conj(H_bc[n+1,n])

					if n<Nx*(Ny-1) and n%2!=0:

						# n = B site to n+Nx = A site
						H_bc[n+Nx,n] = corr_hopping(n+Nx,n,system,'delta_3')
						H_bc[n,n+Nx] = np.conj(H_bc[n+Nx,n])

						if n<Nx*(Ny-1)-1:
							# n+1 = A site to n+1+Nx = B site
							H_bc[n+1,n+1+Nx] = corr_hopping(n+1,n+1+Nx,system,'delta_2')
							H_bc[n+1+Nx,n+1] = np.conj(H_bc[n+1,n+1+Nx])

				if Nx%2==0: # line doesn't begin and end with same site-type
				
					if n%2==0 and (n+1)%Nx!=0:
						# n = B site to n+1 = A site
						H_bc[n+1,n] = honey_hopping(n+1,n,system,'delta_1')
						H_bc[n,n+1] = np.conj(H_bc[n+1,n])

					if n<Nx*(Ny-1) and n%2==0:

						# n = B site to n+Nx = A site
						H_bc[n+Nx,n] = corr_hopping(n+Nx,n,system,'delta_3')
						H_bc[n,n+Nx] = np.conj(H_bc[n+Nx,n])

						if n<Nx*(Ny-1)-1:
							# n+1 = A site to n+1+Nx = B site
							H_bc[n+1,n+1+Nx] = corr_hopping(n+1,n+1+Nx,system,'delta_2')
							H_bc[n+1+Nx,n+1] = np.conj(H_bc[n+1,n+1+Nx]) 

def construct_H_KV(system,case):

	sites_dic = system[0]
	U = system[1]
	Nx = system[3]
	Ny = system[4]

	d1 = np.array([-1,0])
	d2 = np.array([0.5,np.sqrt(3)/2])
	d3 = np.array([0.5,-np.sqrt(3)/2])
	deltas = [d1,d2,d3]

	H_KV = 1j*np.zeros((Nx*Ny,Nx*Ny))	

	if case[2]=='Isotropic': 

		for n in range(Nx*Ny-1):

			H_KV[n,n] = trap(system,case)[n]-U/2

			if n%(2*Nx)<Nx: # line beginning with A site

				if n%2!=0 and (n+1)%Nx!=0: # not on the right boarder
					# n = B site to n+1 = A site
					H_KV[n+1,n] = honey_hopping(n+1,n,system,'delta_1') #link t1
					H_KV[n,n+1] = np.conj(H_KV[n+1,n])

				if n<Nx*(Ny-1) and n%2==0:
					# n = A site to n+Nx = B site
					H_KV[n,n+Nx] = honey_hopping(n,n+Nx,system,'delta_2') #link t2
					H_KV[n+Nx,n] = np.conj(H_KV[n,n+Nx])

					if n<Nx*(Ny-1)-1:
						# n+1 = B site to n+1+Nx = A site
						H_KV[n+1+Nx,n+1] = honey_hopping(n+1+Nx,n+1,system,'delta_3') # link t3
						H_KV[n+1,n+1+Nx] = np.conj(H_KV[n+1+Nx,n+1])

			elif n%(2*Nx)>=Nx: # line beginning with B site

				if Nx%2!=0: # line begins and ends with same site-type

					if n%2!=0 and (n+1)%Nx!=0:
						# n = B site to n+1 = A site
						H_KV[n+1,n] = honey_hopping(n+1,n,system,'delta_1')
						H_KV[n,n+1] = np.conj(H_KV[n+1,n])

					if n<Nx*(Ny-1) and n%2!=0:

						# n = B site to n+Nx = A site
						H_KV[n+Nx,n] = honey_hopping(n+Nx,n,system,'delta_3')
						H_KV[n,n+Nx] = np.conj(H_KV[n+Nx,n])

						if n<Nx*(Ny-1)-1:
							# n+1 = A site to n+1+Nx = B site
							H_KV[n+1,n+1+Nx] = honey_hopping(n+1,n+1+Nx,system,'delta_2')
							H_KV[n+1+Nx,n+1] = np.conj(H_KV[n+1,n+1+Nx])

				if Nx%2==0: # line doesn't begin and end with same site-type
				
					if n%2==0 and (n+1)%Nx!=0:
						# n = B site to n+1 = A site
						H_KV[n+1,n] = honey_hopping(n+1,n,system,'delta_1')
						H_KV[n,n+1] = np.conj(H_KV[n+1,n])

					if n<Nx*(Ny-1) and n%2==0:

						# n = B site to n+Nx = A site
						H_KV[n+Nx,n] = honey_hopping(n+Nx,n,system,'delta_3')
						H_KV[n,n+Nx] = np.conj(H_KV[n+Nx,n])

						if n<Nx*(Ny-1)-1:
							# n+1 = A site to n+1+Nx = B site
							H_KV[n+1,n+1+Nx] = honey_hopping(n+1,n+1+Nx,system,'delta_2')
							H_KV[n+1+Nx,n+1] = np.conj(H_KV[n+1,n+1+Nx]) 

			H_KV[-1,-1] = trap(system,case)[-1]-U/2

	return H_KV