#Written by Weihao Gao from UIUC

import scipy.spatial as ss
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers


# Usage functions
def mi(x,y,k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using KSG mutual information estimator

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter

		Output: one number of I(X;Y)
	'''
	assert len(x)==len(y), "Lists should have same length"
   	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
   	dx = len(x[0])   	
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

   	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

   	knn_dis = [tree_xy.query(point,k+1,p=2)[0][k] for point in data]
	ans = digamma(k) + 2*log(N-1) - digamma(N) + vd(dx) + vd(dy) - vd(dx+dy)
	for i in range(N):
		ans += -log(len(tree_x.query_ball_point(x[i],knn_dis[i],p=2))-1)/N 
		ans += -log(len(tree_y.query_ball_point(y[i],knn_dis[i],p=2))-1)/N 
		
	return ans


def umi(x,y,k=5,bw=0.2):
	'''
		Estimate the uniform mutual information UMI(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using UMI estimator in http://arxiv.org/abs/1602.03476

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		bw: bandwidth selection of KDE

		Output: one number of UMI(X;Y)
	'''
	assert len(x)==len(y), "Lists should have same length"
   	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
   	dx = len(x[0])   	
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)
	
	kernel = KernelDensity(bandwidth = bw)
	kernel.fit(x)
	kde = np.exp(kernel.score_samples(x))
	weight = (1/kde)/np.mean(1/kde)
	
   	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

   	knn_dis = [tree_xy.query(point,k+1,p=2)[0][k] for point in data]
	ans = digamma(k) + 2*log(N-1) - digamma(N) + vd(dx) + vd(dy) - vd(dx+dy)
	for i in range(N):	
		nx = len(tree_x.query_ball_point(x[i],knn_dis[i],p=2))-1
		ny = np.sum(weight[j] for j in tree_y.query_ball_point(y[i],knn_dis[i],p=2)) - weight[i]
		ans += -weight[i]*log(nx)/N 
		ans += -weight[i]*log(ny)/N 		
	return ans


def cmi(x,y,k=5,bw=0.2,init_weight_option=1,eta=0.5,lamb=0.5,T=500):
	'''
		Estimate the capacitated mutual information CMI(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using CMI estimator in http://arxiv.org/abs/1602.03476

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		bw: bandwidth selection of KDE
		init_weight_option: initialization of w, 0=uniform, 1=true density
		eta: step size of SGD
		lambda: coefficient of smoothness regularizer
		T: number of steps in SGD

		Output: one number of CMI(X;Y)
	'''
	
	assert len(x)==len(y), "Lists should have same length"
   	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	solvers.options['show_progress'] = False
	N = len(x)
   	dx = len(x[0])   	
	dy = len(y[0])
	y = y[np.argsort(x,axis=0)].reshape(N,1)
	x = np.sort(x,axis=0)
	data = np.concatenate((x,y),axis=1)
	
	if init_weight_option == 0:
		weight = np.ones(N)+nr.normal(0,0.1,N)
	else:
		kernel = KernelDensity(bandwidth = bw)
		kernel.fit(x)
		kde = np.exp(kernel.score_samples(x))
		weight = (1/kde).clip(1e-8,np.sqrt(N))
		weight = weight/np.mean(weight)

	A = np.zeros(N)
	b = 0
	for i in range(N):
		A[i] = (x[i]-np.mean(x))**2
		b += weight[i]*A[i]
	
   	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

   	knn_dis = [tree_xy.query(point,k+1,p=2)[0][k] for point in data]
	adj_x = []
	adj_y = []
	for i in range(N):
		adj_x.append(tree_x.query_ball_point(x[i],knn_dis[i],p=2))
		adj_y.append(tree_y.query_ball_point(y[i],knn_dis[i],p=2))
		
	ans = digamma(k) + 2*log(N-1) - digamma(N) + vd(dx) + vd(dy) - vd(dx+dy)
	
	for i in range(T):
		ind = nr.randint(N)
		gradient = get_grad(adj_x,adj_y,weight,ind)
		weight = weight + eta*(gradient-lamb*d_regularizer(weight))
		weight = projection(weight,A,b)

	return ans+get_obj(adj_x,adj_y,weight)
	

#Auxilary Functions

def vd(d):
	# Return the volume of unit ball in d dimension space
	return 0.5*d*log(pi) - log(gamma(0.5*d+1))

def entropy(x,k=5):
	# Estimator of (differential entropy) of X 
	# Using k-nearest neighbor methods
   	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
   	d = len(x[0]) 	
   	tree = ss.cKDTree(x)
   	knn_dis = [tree.query(point,k+1,p=2)[0][k] for point in x]
	ans = -digamma(k) + digamma(N) + vd(d)
	return ans + d*np.mean(map(log,knn_dis))

# Following functions are used in SGD to compute CMI. 
def d_regularizer(weight):
	# Compute the derivative of smoothness regularizer
	N = len(weight)
	ans = np.zeros(len(weight))
	for i in range(len(weight)-1):
		ans[i] += weight[i]-weight[i+1]
		ans[i+1] += weight[i+1]-weight[i]
	return ans/N
		
def get_obj(adj_x,adj_y,weight):
	# Compute the objective function
	N = len(adj_x)
	ans = 0
	for i in range(N):
		nx = len(adj_x[i])-1
		ny = np.sum(weight[j] for j in adj_y[i]) - weight[i]
		ans += -weight[i]*log(nx)/N
		ans += -weight[i]*log(ny)/N 		
	return ans

def get_grad(adj_x,adj_y,weight,i):
	# Compute the gradient w.r.t. w_i
	N = len(adj_x)
	ans = np.zeros(N)
	nx = len(adj_x[i])-1
	ny = np.sum(weight[j] for j in adj_y[i]) - weight[i]
	for j in adj_y[i]:
		ans[j] += -weight[i]/(ny*N)
	ans[i] += -(log(nx)+log(ny))/N +weight[i]/(ny*N)
	return ans*np.sqrt(N)

def projection(w,A,b):
	# Projection to the constrained space
	N = len(w)
	w = w.reshape((len(w),1))
	A = A.reshape((len(w),1))
	x = np.dot(np.linalg.inv(np.dot(A.transpose(),A)),np.dot(A.transpose(),w)-b)
	if x>0:
		w = w - np.dot(A,x)
	w = w.clip(1e-6).reshape(N)
	return w/np.mean(w)

if __name__ == '__main__':
	print "Please read readme.pdf for guidance"
