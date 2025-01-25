# Bayesian-stats-project

# Reproduction and Extension of Article 3: Optimal Bayesian Estimation of Gaussian Mixtures with a Growing Number of Components

## Introduction

This project aims to reproduce and extend the work presented in the following article:

**Ohn, I. & Lin, L. (2023). Optimal Bayesian estimation of Gaussian mixtures with growing number of components. _Bernoulli_, 29(2), 1195–1218. [DOI](https://doi.org/10.3150/22-BEJ1495)**

The primary objective is to validate the theoretical results of the article by implementing the proposed methods in Python and to explore potential extensions to enhance or generalize these findings.

## Project Objectives

1. **Reproduction of Results**: Implement the algorithms described in the article to estimate Gaussian mixtures with a growing number of components. Validate the theoretical results through simulation studies.

2. **Extension of Methods**: Propose and implement improvements or generalizations of the original methods. This may include applying the methods to more complex mixture models, optimizing computational performance, or exploring new practical applications.

## Project Structure

The project is organized as follows:

- **Documentation**
  - `README.md`: This documentation file.
  - `Garnier_Clayton_Gimenes_Project.pdf`: Final report of approximately 5 pages, without appendix, summarizing the work performed and the results obtained.
  - `Article_lin.pdf` : "Optimal Bayesian Estimation of Gaussian Mixtures with a Growing Number of Components” by Ohn and Lin

- **Source Code**
  - `Code/`: Contains all the Python scripts necessary for reproducing and extending the results, and jupyter notebooks for interactive data exploration and result visualization.

    `bayes.py` : RJMCMC sampler for Bayesian mixtures with hierarchical priors
    `discrete_rv.py` : module for common operations on discrete random variables (distributions)
    `dmm.py` : DMM algorithm of Wu and Yang (2020, Ann. Stat.)
    `DP.py` : Neal’s Algorithm 8 for DP mixtures
    `em_bayes.py` : EM algorithm for obtaining posterior mode of a Bayesian mixture given the number of components and model selection
    `model_gm.py` : module for Gaussian mixture model
    `model_student_mixture.py` : module for Student mixture model
    `moments.py` : module for operations on moments
    `Notebook_merged_gaussian.ipynb` : Superposition of every useful notebook to get results in the gaussian mixture model
    `Notebook_merged_tstudent.ipynb` : Superposition of every useful notebook to get results in the student t mixture model
    `test_mixing` : Gaussian mixing distribution estimation


  - `Pictures/`: Contains visualizations of different wassertstein distance depending on models and their parametrization.

## Prerequisites

To run the project code, ensure you have the following software and libraries installed:

- **Programming Language**: Python (version 3.8 or higher)
- **Python Libraries**:
  - `numpy`
  - `scipy`
  - `maths`
  - `moments`
  - `discrete_rv`
  - `matplotlib`
  - `cvxpy`
  - `random`

- **Other Tools**:
  - Jupyter Notebook
  - Git

## Installation

1. Clone the Project Repository:
```bash
   git clone https://github.com/Pierre-Clayton/Bayesian-stats-project.git
   cd Bayesian-stats-project
```

2.	Create a Virtual Environment (Recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3.	Install Required Libraries:
```bash
pip install -r requirements.txt
```

Example of running a Python script:
```bash
python src/estimate_mixture.py
```
Contributions

This project is carried out by a group of three students:
	•	Student 1: Pierre CLAYTON 
	•	Student 2: Thomas GARNIER 
	•	Student 3: Vincent GIMENES


