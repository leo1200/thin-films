import jax
import jaxopt
import jax.numpy as jnp
import jax.scipy.optimize
from reflax.data_analysis.chirpgp.models import g, g_inv, build_chirp_model
from reflax.data_analysis.chirpgp.filters_smoothers import ekf, eks
from reflax.data_analysis.chirpgp.quadratures import gaussian_expectation

# Variant 1
def chirp_analyzer(chirp_signal, dt, noise_std):
    # MLE parameter estimation
    # From left to right, they are, lam, b, delta, ell, sigma, m0_1
    init_theta = g_inv(jnp.array([0.1, 0.1, 0.1, 1., 1., 7.]))

    # Objective function
    @jax.jit
    def obj_func(theta: jnp.ndarray):
        _, _, m_and_cov, m0, P0, H = build_chirp_model(g(theta))
        return ekf(m_and_cov, H, noise_std, m0, P0, dt, chirp_signal)[-1][-1]

    # Optimise
    opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
    opt_vals, opt_state = opt_solver.run(init_theta)
    opt_params = g(opt_vals)
    print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

    # Filtering and smoothing based on the learnt parameters
    _, _, m_and_cov, m0, P0, H = build_chirp_model(opt_params)

    @jax.jit
    def filtering(measurements):
        return ekf(m_and_cov, H, noise_std, m0, P0, dt, measurements)

    @jax.jit
    def smoothing(mfs, Pfs):
        return eks(m_and_cov, mfs, Pfs, dt)

    # Trigger jit
    _dummy = filtering(jnp.ones((2,)))
    smoothing(_dummy[0], _dummy[1])

    filtering_results = filtering(chirp_signal)
    smoothing_results = smoothing(filtering_results[0], filtering_results[1])

    # Note that the distribution of f=g(V) is not Gaussian
    # The confidence interval in the following may not be centred at E[g(V)]
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, 2],
                                                chol_Ps=jnp.sqrt(smoothing_results[1][:, 2, 2]),
                                                func=g, force_shape=True)[:, 0]
    
    return estimated_freqs_mean, smoothing_results



# VARIANT 2
# def chirp_analyzer(chirp_signal, dt, noise_std):
#     # Sigma points
#     sgps = SigmaPoints.gauss_hermite(d=4, order=3)

#     # MLE parameter estimation
#     # From left to right, they are, lam, b, delta, ell, sigma, m0_1
#     init_theta = g_inv(jnp.array([0.1, 2., 0.5, 0.02, 40., 1.]))


#     # Objective function
#     def obj_func(theta: jnp.ndarray):
#         _, _, m_and_cov, m0, P0, H = build_chirp_model(g(theta))
#         return sgp_filter(m_and_cov, sgps, H, noise_std, m0, P0, dt, chirp_signal)[-1][-1]


#     # Optimise
#     opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
#     opt_vals, opt_state = opt_solver.run(init_theta)
#     opt_params = g(opt_vals)
#     print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

#     # Filtering and smoothing based on the learnt parameters
#     _, _, m_and_cov, m0, P0, H = build_chirp_model(opt_params)


#     @jax.jit
#     def filtering(measurements):
#         return sgp_filter(m_and_cov, sgps, H, noise_std, m0, P0, dt, measurements)


#     @jax.jit
#     def smoothing(mfs, Pfs):
#         return sgp_smoother(m_and_cov, sgps, mfs, Pfs, dt)


#     # Trigger jit
#     _dummy = filtering(jnp.ones((1,)))
#     smoothing(_dummy[0], _dummy[1])

#     # Compute posterior distributions
#     filtering_results = filtering(chirp_signal)
#     smoothing_results = smoothing(filtering_results[0], filtering_results[1])
#     estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, 2],
#                                                 chol_Ps=jnp.sqrt(smoothing_results[1][:, 2, 2]),
#                                                 func=g, force_shape=True)[:, 0]
    
#     return estimated_freqs_mean, smoothing_results