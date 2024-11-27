import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

from DP import *
from bayes import *                
                

n_save = 1000
n_list = [50, 100, 250, 1000, 2500]

k_list =[]
k_list = np.zeros([4, len(n_list), n_save])

for i, n in enumerate(n_list):
    x =  np.random.choice([-3, 0, 3], size=n) + norm.rvs(size=n)        
    _, _, k = bayesGM(x, interval=[-6,6], lam=10*np.exp(-0.2*np.log(n)**2/np.log(np.log(n))), kappa=1, n_save=n_save, n_burn=5000, n_thin=100)
    k_list[0,i,:] = k
    _, _, k = bayesGM(x, interval=[-6,6], lam=1, kappa=1, n_save=n_save, n_burn=5000, n_thin=100)
    k_list[1,i,:] = k
    _, k = DP(x, interval=[-6,6], kappa=20/n, n_save=n_save, n_burn=200, n_thin=5)
    k_list[2,i,:] = k
    _, k = DP(x, interval=[-6,6], kappa=0.4, n_save=n_save, n_burn=200, n_thin=5)
    k_list[3,i,:] = k
    
## Plot
markers=['^','o','s','D','*']
methods = ["bayes_", "dp_"]
hyperpars = ["vary", "const"]

for a in range(4):
    plt.figure() 
    for m in range(len(k_list[a])):
        k = np.append(k_list[a,m,:], np.arange(10)+1 )
        px, py = np.unique(k, return_counts=True)    
        mark = markers[m]
        plt.plot(px,py/1000, marker=mark, label='n= %s' % n_list[m]) 
    if a==0:
        plt.legend()
    plt.ylim(0, 1)
    plt.xlabel("Number of components")
    plt.ylabel("Posterior probability")
    plt.savefig("ncomp_"+methods[a//2]+hyperpars[a%2]+".png")
