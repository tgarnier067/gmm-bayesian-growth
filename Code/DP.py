import numpy as np
#from nn_model_fts import *
from scipy.stats import norm
from scipy.stats import truncnorm


def DP(x, interval, kappa, n_save, n_burn, n_thin):
    
    n_iter = n_burn + n_save*n_thin    
    k_mcmc = []
    theta_mcmc = []
    
    n = len(x)
    K_new=5
    ## initialize
    theta = np.random.uniform(interval[0], interval[1], size=6) 
    z = np.random.choice(np.arange(6)+1, size=n)
    
    for t in range(n_iter):
        
        for i in range(n):
            T = np.max(z)
            z_i = np.delete(z, i)
            theta_new = np.random.uniform(interval[0], interval[1], size=K_new)             
            
            unnorm_prob = np.zeros(T+K_new)
            for j in range(T+K_new):
                if j < T:
                    unnorm_prob[j] = np.sum(np.equal(z,j+1))*norm.pdf(x[i], loc=theta[j])   
                else:
                    unnorm_prob[j] = kappa*norm.pdf(x[i], loc=theta_new[j-T])/K_new      
                    
            prob =  unnorm_prob/np.sum(unnorm_prob)
            z_new = np.random.choice(np.arange(T+K_new) + 1, p=prob)
            
            if np.sum(np.equal(z_i, z[i]))>0: 
                if z_new > T:
                    z[i] = T + 1 
                    theta = np.append(theta, theta_new[z_new-T-1])
                else:
                    z[i] = z_new 
                    
            else: 
                if z_new > T:
                    theta[z[i]-1] = theta_new[z_new-T-1]
                else:
                    if z_new !=z[i]:
                        z_old = z[i]
                        z[i] = z_new
                        theta = np.delete(theta, z_old-1)                        
                        z = z - (np.array(z)>z_old)                             
            
        T = np.max(z)    
        for j in range(T):
            x_j = x[np.equal(z, j+1)]
            theta[j] = truncnorm.rvs(interval[0], interval[1], loc=np.mean(x_j), scale=1/np.sqrt(len(x_j)))      
                
        if (t+1>n_burn) & ((t+1-n_burn)%n_thin==0):
            theta_mcmc.append(theta)
            k_mcmc.append(len(np.unique(z)))
            
    return theta_mcmc,  k_mcmc