# reflax - Differentiable Interference Modeling for Cost-Effective Growth Estimation of Thin Films

![reflax](logo.svg)
 
> The results from the paper can be reproduced using the scripts in the `paper` folder.

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


## Installation


## Forward modeling: thickness time series → reflectance time series

### Setting up the forward model

### Generating an examplary thickness time series

### Running the forward model


## Inverse modeling: reflectance time series → thickness time series

In the following, we will show how you can from a single wavelength reflectance time series of a growing thin film infer the thickness time series.

> Note that here we consider inference under a given experimental setup with known refractive index of the thin layer. While the refractive index could also be inferred, we have not yet adapted our pipeline accordingly.

### Training a neural operator for inferring the thickness time series

> Note that the aforementioned neural operator initialization requires a known given total duration of the growth process and fixed experimental setup - but the neural operator can be trained very quickly (sub 30 seconds on an NVIDIA A100) so this is not much of a limitation.


### Inferring the thickness time series by optimization through the differentiable simulator