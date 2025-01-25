# Optimal Bayesian estimation of Gaussian mixtures with growing number of components

Python code for replicating and deepen the simulation results in the paper "Optimal Bayesian estimation of Gaussian mixtures with growing number of components" [[arXiv]](https://arxiv.org/abs/2007.09284). This work explores practical implications of replacing Gaussian components with t-Student components.

## Experiments

* Gaussian mixing distribution estimation: `test_mixing.py`
* Student t mixing distribution estimation: `Notebook_merged_student.ipynb`


## Methods

* RJMCMC sampler for Bayesian mixtures with hierarchical priors: `bayes.py`
* EM algorithm for obtaining posterior mode of a Bayesian mixture given the number of components and model selection `em_bayes.py`
* Neal’s Algorithm 8 for DP mixtures `DP.py`
* DMM algorithm of Wu and Yang (2020, Ann. Stat.) `dmm.py`


### Disclaimer

The python files `bayes.py`, `discrete_rv.py`, `dmm.py`, `DP.py`, `model_gm.py`, `moments.py` and `test_mixing.py` are taken from [https://github.com/ilsangohn/bayes_mixture](https://github.com/ilsangohn/bayes_mixture)