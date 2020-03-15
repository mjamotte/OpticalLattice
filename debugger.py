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

################### DEBUGGER ###################

def debug_plot_vector(vector_s,args_plot):

	# vector_s contains same type of vectors

	params_fig = args_plot['params_fig']
	
	# vector is a simple array
	if type(vector_s[0])==type(np.array([1.618])[0]) or \
		type(vector_s[0])==type(np.array([1j*1.618])[0]):
		if 'abscisse' in args_plot:
				abscisse = args_plot['abscisse']
				fig = pyplot.plot(abscisse,vector)
		else:
			fig = pyplot.plot(np.arange(len(vector_s)),vector_s)
	# vector is a array of arrays
	elif type(vector_s[0])==type(np.array([1.618])):
		for vector in vector_s:
			if 'abscisse' in args_plot:
				abscisse = args_plot['abscisse']
				fig = pyplot.plot(abscisse,vector)
			else:
				fig = pyplot.plot(np.arange(len(vector)),vector)
	else:
		fig = 'Type of input "vector" unrecognised'
		

	if 'xlabel' in params_fig:
		pyplot.xlabel(params_fig['xlabel'])

	if 'ylabel' in params_fig:
		pyplot.ylabel(params_fig['ylabel'])

	pause = params_fig['pause']

	pyplot.draw()
	pyplot.pause(pause)

	return fig

def debug_values(variable,args_values):

	if 'precision' in args_values:
		prec = args_values['precision']
		np.set_printoptions(precision=prec)

	print(variable)

