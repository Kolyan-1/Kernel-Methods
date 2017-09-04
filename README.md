#MSc Thesis Code by Nikolai Rozanov (2017)

This code is a small library to perform statistical tests with Kernel Independence measures.
So far the implementation only uses numpy and scipy (i.e. standard python libraries)

## Procedures in this Library:

* Various Kernels (public fields: parameters, calculations of Kernel Matrices)

* HSIC estimators (for iid data only so far)

* Various Statistics Estimators (HSIC Test quantile Estimator, HSIC power Estimator, both quadratic time/linear space and linear time/constant space)

* Main Procedures with the above:

    1. Learning the Kernel

    2. Evaluating the test

## Requirements

To install requirements run 'pip install -r requirements.txt'
