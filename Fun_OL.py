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


def density(case,system,H_KV):

	if case[0]=='SC': #SC = self-consistent method
		E0,psi_0,E0_all,psi0_all,E_funct = solveGP_SC(case,system,H_KV)
		out = np.array([E0,abs(psi_0)**2,E0_all,abs(psi0_all)**2,E_funct])

	elif case[0]=='IT': #Imaginary time
		E_all,psi_all = solveGP_IT(case,system)
		out = np.array([E_all,abs(psi_all)**2])
	
	return out


def solveGP_SC(case,system,H_KV):

	N = system[2]
	E_funct = np.array([])
	E0_all = [np.zeros(len(H_KV))] # to observe convergence of the lowest energy
	psi0_all = [np.array([])]
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap	

	#E0_temp1,eigVecs = sc.sparse.linalg.eigsh(H_KV,which='SA',k=1)
	E0_temp1,eigVecs = linalg.eigh(H_KV)
	psi0_temp = np.matrix.transpose(eigVecs[:,0])
	#psi0_temp = impose_sym(psi0_temp)

	E0_all = [E0_temp1]
	psi0_all = np.append(psi0_all,psi0_temp)

	counterSC = 0
	lam = 1 #1/(10*N+1) # to avoid oscillations in energy that slow down the code
	flag = 0
	err = 1e-9

	while True:

		#H_U = sc.sparse.coo_matrix(0.5*U*np.diag(np.abs(psi0_temp)**2))
		#E0_temp2,eigVecs = sc.sparse.linalg.eigsh(H_KV+H_U,which='SA',k=1)
		H_U = H_int(N,psi0_temp)
		E0_temp2,eigVecs = linalg.eigh(H_KV+H_U)
		psi0_temp1 = np.matrix.transpose(eigVecs[:,0])
		#psi0_temp1 = impose_sym(psi0_temp1)

		psi0_lam = np.sqrt((1-lam)*psi0_temp**2 + lam*psi0_temp1**2)

		E_funct = np.append(E_funct,energy_functional(psi0_lam,system,case,E0_temp2[0]))

		if len(E_funct)>2:
			if E_funct[-1]>E_funct[-2] and flag==0:
				flag = 1
				print('Energy functional is not decreasing')

		E0_all = np.append(E0_all,[E0_temp2],axis=0)
		psi_all = np.append(psi0_all,psi0_lam,axis=0)

		#if abs((E0_temp1[0]-E0_temp2[0])/E0_temp1[0])<err:
		if np.sum((np.abs(psi0_temp)**2-np.abs(psi0_lam)**2)**2)<err:
			break

		psi0_temp = psi0_lam
		E0_temp1 = E0_temp2
		counterSC += 1			

	print('Number of interations of self-consistent method =',counterSC)
	return E0_temp2,psi0_temp,E0_all,psi_all,E_funct


def H_int(N,psi):
	g = 1e-2
	return g*N*np.diag(np.abs(psi)**2)

def energy_functional(psi,system,case,mu):

	J = system[1]
	N = system[2]
	V0 = system[3]
	Nx = system[4]
	g = 0.01

	E_U = g*N/2*np.sum(np.abs(psi)**4)-g*N/2
	E_trap = np.sum(trap_1D(system,case)*abs(psi)**2)
	
	positions = np.arange(Nx-1) # -1 because of the k+1 below
	E_kin = 0
	for k in positions:
		E_kin += -J*np.conj(psi[k])*psi[k+1]-J*np.conj(psi[k+1])*psi[k]

	return E_U+E_trap+E_kin#-mu*N # = <H>/N

#@jit
def GP(t,psi_old,system,H_KV):

	N = system[2]
	dim = len(psi_old)
	psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]

	# Hopping part + trap part of the GP
	y1 = H_KV.dot(psi_old_co)
	# Interacting part of the GP
	y2 = H_int(N,psi_old_co).dot(psi_old_co)

	y = y1 + y2
	# -d_tau psi(tau) = H psi(tau)
	y = -y
	return np.concatenate((np.real(y),np.imag(y)))


def impose_sym(vector): 
# imposes symmetry from the middle of the vector
	Len = len(vector)

	for i in range(int(Len/2)):
		vector[i] = (vector[i]+vector[Len-i-1])*0.5
		vector[Len-i-1] = vector[i]

	norm = np.sqrt(np.sum(np.abs(vector)**2))
	vector = vector/norm
	return vector


############# 1D OPTICAL LATTICE #############

''' 
	The goal of coding this is the understanding of the physics 
	of the Hamiltonian presented in the Lühmann's paper to 
	aquire feelings with correlated hoppings. 
'''

def lattice_1D(Nx):

	sites_dic = {}
	n = 0
	for i in range(Nx): # fill line by line, from left to right
		sites_dic[n] = np.array([n])
		n += 1
			
	return sites_dic

	return 	

def trap_1D(system,case):

	V0 = system[3]
	Nx = system[4]

	V = np.zeros(Nx)

	for x in range(Nx):
		V[x] = 1/2*V0*(x-(Nx-1)/2)**2

	return V


def H_1D(system,case):

	J = system[1]
	Nx = system[4]

	H = 1j*np.zeros((Nx,Nx))

	if case[2]=='Isotropic':

		if case[1]=='Harmonic':
			H = np.diag(trap_1D(system,case))

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