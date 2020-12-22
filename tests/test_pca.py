from spectrophotometricpca.pca import *

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_prior():

    num_components = 3
    nobj = 10
    prior = PriorModel(num_components)

    params = prior.random(key)
    assert_shape(params, prior.params_shape)

    redshifts = 10 ** jax.random.normal(key, (nobj,))
    mu = prior.get_mean_at_z(params, redshifts)
    assert_shape(mu, (nobj, num_components))
    muloginvvar = prior.get_loginvvar_at_z(params, redshifts)
    assert_shape(muloginvvar, (nobj, num_components))
