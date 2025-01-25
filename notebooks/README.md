# Optimal Bayesian estimation of Gaussian mixtures with growing number of components

Python code for replicating the simulation results in the paper "Optimal Bayesian estimation of Gaussian mixtures with growing number of components" [[arXiv]](https://arxiv.org/abs/2007.09284)


## Experiments

* Mixing distribution estimation: `test_mixing.py`
* The number of components estimation: `test_ncomp.py`

## Methods

* RJMCMC sampler for Bayesian mixtures with hierarchical priors: `bayes.py`
* EM algorithm for obtaining posterior mode of a Bayesian mixture given the number of components and model selection `em_bayes.py`
* Neal’s Algorithm 8 for DP mixtures `DP.py`
* DMM algorithm of Wu and Yang (2020, Ann. Stat.) `dmm.py`


### Disclaimer

The python files `discrete_rv.py`, `dmm.py`, `model_gm.py` and `moments.py` are taken from [https://github.com/albuso0/mixture](https://github.com/albuso0/mixture).
