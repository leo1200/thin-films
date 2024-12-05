# Introduction and Installation

`reflax` is a toolbox for forward-modeling and inference for interference on
growing thin films in a multi-layer setup written in `JAX`.


::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Fast and Differentiable

Written in `JAX`, `reflax` is fully differentiable - a simulation can be differentiated with respect to any input parameter - and just-in-time compiled for fast execution on CPU, GPU, or TPU.

+++
[Learn more »](notebooks/one_growing_layer.ipynb)
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` All-in-one solution

`reflax` includes all the inference tools necessary to calibrate optical parameters
and infer growth behavior from experimental data. Based on the differentiability of the
forward models implemented, [Levenberg Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) can be employed for optimization.

+++
[Learn more »](notebooks/one_growing_layer.ipynb)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Easily Adaptable

You want to infere based on the reflectance signal of multiple LASERs? You want to
train a machine learning model to predict the growth rate based on environmental
parameters? - adapting the loss or training a machine learning model inside `reflax`
is straightforward.

+++
[Learn more »](notebooks/one_growing_layer.ipynb)
:::

::::


## Installation

`reflax` can be installed via `pip`

```bash
pip install git+https://leo1200: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@github.com/leo1200/thin-films.git#v0.0.3
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

:::{tip} Get started with this [simple example](notebooks/one_growing_layer.ipynb).
:::

## Roadmap

- [x] implement in `JAX`
- [ ] detailed model comparison
- [ ] train an SBI pipeline, e.g. a `simformer`


```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

self
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Notebooks

notebooks/one_growing_layer.ipynb
```

```{toctree}
:hidden:
:caption: Reference

source/reflax.transfer_matrix_method.transfer_matrix_method
```