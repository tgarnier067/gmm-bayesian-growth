import numpy as np
import math

from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import dirichlet

def log_lkd(x, w, theta):
    """
    log-likelihood of samples

    Args:
    x: samples
    w: weight vector
    theta: centers
    """
    k = len(w)
    n = len(x)
    lkd_mtx = np.ones((n,k))
    for j in range(k):
        lkd_mtx[:,j] = w[j]*norm.pdf(x, loc=theta[j])         
    lkd = np.sum(lkd_mtx, axis=1)
    return np.sum(np.log(lkd))

def log_prior(w, interval, lam, kappa):
    """
    log of prior density

    Args:
    w: weight vector
    interval: range of centers
    lam: mean parameter of  Poisson prior
    kappa: hyperparameter for Dirichlet prior
    """
    k = len(w)
    k_p = poisson.logpmf(k-1, lam)
    if k>1:
        w_p = dirichlet.logpdf(w, kappa*np.ones(k))  
    else:
        w_p = 0
    theta_p = -k*np.log(interval[1]-interval[0])
    return k_p + w_p + theta_p + math.lgamma(k+1)


def bayesGM(x, interval, lam, kappa, n_save, n_burn, n_thin):
    """
    RJMCMC sampler for Bayesian mixture

    Args:
    x: samples
    interval: range of centers
    lam: mean parameter of  Poisson prior
    kappa: hyperparameter for Dirichlet prior
    n_save, n_burn, n_thin: numbers of saved, burn-in and thinned MCMC samples, resp.
    """
    
    n_iter = n_burn + n_save*n_thin    
    k_mcmc = []
    theta_mcmc = []
    w_mcmc = []  
    
    ## initialize
    w = dirichlet.rvs(np.ones(6))[0]
    theta = np.random.uniform(interval[0], interval[1], size=6) 
    oll = log_lkd(x, w, theta)
    olp = log_prior(w, interval, lam, kappa)
    
    for t in range(n_iter):
        
        OK = True
        move = np.random.choice([1,2,3,4])
        k = len(w)
        
        if move==1: # add a component   
            j = np.random.choice(np.arange(k))
            w_s = np.random.uniform(0, w[j], size=1)[0]
            w1 = np.append(w, w_s)
            w1[j] = w[j] - w_s                     
            theta_s = np.random.uniform(interval[0], interval[1], size=1)[0]
            theta1 = np.append(theta, theta_s)
            log_rho_d = -np.log(k)
            log_rho_n = -np.log(k*(k+1))
            log_q_d = -np.log(interval[1]-interval[0]) - np.log(w1[j])
            log_q_n = 0
        
        if move==2: #delete a component     
            if k > 1:
                jh = np.random.choice(np.arange(k), 2, replace=False)
                j = jh[0]
                h = jh[1]
                w1 = w.copy()
                w1[j] = w[j] + w[h]                
                w1 = np.delete(w1, h)                
                theta1 = np.delete(theta, h) 
                log_rho_d = -np.log(k*(k-1))
                log_rho_n = -np.log(k-1)
                log_q_d = 0
                log_q_n = -np.log(interval[1]-interval[0]) - np.log(w[j]+w[h])
            else:
                OK = False
        
        if move==3: # sample theta
            j = np.random.choice(np.arange(k))
            theta1 = theta.copy()
            theta1[j] = np.random.uniform(interval[0], interval[1], size=1)[0]
            w1 = w.copy()
            log_rho_d = log_rho_n = log_q_d = log_q_n = 0        
        
        if move==4: #sample w
            if k > 1:
                jh = np.random.choice(np.arange(k), 2, replace=False)
                j = jh[0]
                h = jh[1]
                w1 = w.copy()
                w1[j] = np.random.uniform(0, w[j] + w[h], size=1)[0]
                w1[h] = w[j] + w[h] - w1[j]
                theta1 = theta.copy()
                log_rho_d = log_rho_n = log_q_d = log_q_n = 0    
            else:
                OK = False          
                
        if OK:            
            nll = log_lkd(x, w1, theta1)
            nlp = log_prior(w1, interval, lam, kappa)            
            MHR = nll + nlp - oll - olp + log_rho_n - log_rho_d + log_q_n - log_q_d 
            
            if np.log(np.random.uniform(size=1))<MHR:
                w = w1.copy()
                theta = theta1.copy()
                oll = nll.copy()
                olp = nlp.copy()              
                
        if (t+1>n_burn) & ((t+1-n_burn)%n_thin==0):
            theta_mcmc.append(theta)
            w_mcmc.append(w)
            k_mcmc.append(len(w))
            
    return theta_mcmc, w_mcmc, k_mcmc
