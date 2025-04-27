"""

The problem to solve: Optimization through the simulator can be slow
as of the complex loss. For broad adoption of our method, quickly
finding the growth behavior is key. Using that the phase is related
to the growth by determining the instantaneous frequency (Hilbert
transform, ChirpGP) is often not feasible as we only have a few periods
and our signal form is very specific (one might be able to generalize
ChirpGP to different priors though).

Solution: Train a neural operator to infer the growth behavior and
tackle the details using optimization through the simulator.

Reduction to a minimal viable product: We will only consider one baseline
simulation setup, consider depositions of one hour with average growth of
800 nm/h up to 1000 nm/h. We will use a fixed time resolution.

"""
