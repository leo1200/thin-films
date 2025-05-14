# reflax - Differentiable Interference Modeling for Cost-Effective Growth Estimation of Thin Films

> [!NOTE]  
> The results from the paper can be reproduced using the scripts in the `paper` folder. We are still in the process of cleaning up these scripts until the supplemental material deadline (May 22 '25 (Anywhere on Earth)).

`reflax` is an open-source `JAX` library for inferring thin film growth behavior, featuring differentiable interference forward models

- a single layer ray propagation model
- a general transfer matrix model

to go from thin layer thickness and other setup parameters to reflectance as well as

- a predefined neural network architecture outputting a monotonic thickness function (thickness model)
- an optimization loop to optimize the parameters of the thickness model to minimize the discrepency between the simulator output and a mea

and

- function sampling tools to generate example thickness functions (based on Gaussian process samples) with user specified bounds on the final thickness as well as the minimum and maximum growth rates

which can be used to generate training data for

- a neural operator model estimating the a thickness time series from a reflectance time series

which may be used as an initialization in the optimization of the thickness model through the differentiable interference forward model.

## Short Documentation
TODO: add short documentation here