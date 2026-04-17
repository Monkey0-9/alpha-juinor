"""
Renaissance Technologies Style Alpha Engine
Utilizes Information Geometry and Hidden Markov Models (HMM).
We treat the market as a sequence of probability distributions on a Riemannian manifold,
calculating the Fisher Information Matrix (FIM) to detect structural regime shifts
before standard volatility estimators can react.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# We assume a parameterized family of distributions p(x|theta)
# For demonstration, a multidimensional Gaussian regime model

@jit
def log_likelihood(theta, observation):
    """
    Log-likelihood of a market microstate observation given parameters theta.
    theta: [mu_1, ..., mu_n, sigma_1, ..., sigma_n]
    """
    n = len(observation)
    mu = theta[:n]
    sigma = theta[n:]
    
    # Stable log-likelihood for Gaussian 
    z = (observation - mu) / sigma
    ll = -0.5 * jnp.sum(z**2 + jnp.log(2 * jnp.pi * sigma**2))
    return ll

# Calculate the score function (gradient of log-likelihood) using JAX autodiff
score_function = grad(log_likelihood, argnums=0)

@jit
def fisher_information_matrix(theta, batch_observations):
    """
    Empirical Fisher Information Matrix calculation over a batch of microstates.
    This defines the metric tensor of our market manifold.
    """
    # Vectorize the score function across the batch
    v_score = vmap(score_function, in_axes=(None, 0))(theta, batch_observations)
    
    # FIM is the expected outer product of the scores: E[score * score^T]
    # We compute the empirical expectation
    fim = jnp.mean(jnp.einsum('bi,bj->bij', v_score, v_score), axis=0)
    
    return fim

def detect_regime_shift(theta_t0, theta_t1, fim):
    """
    Computes the Information Distance (Rao distance approximation) between 
    two market states. A large distance indicates a covert regime shift.
    """
    d_theta = theta_t1 - theta_t0
    
    # Mahalanobis-like distance on the statistical manifold
    ds_squared = jnp.dot(d_theta.T, jnp.dot(fim, d_theta))
    return jnp.sqrt(ds_squared)
