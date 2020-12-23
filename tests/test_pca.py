from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_chebychevPolynomials():
    numPoly, numSpecPix = 3, 100
    res = chebychevPolynomials(numPoly, numSpecPix)
    assert_shape(res, (numPoly, numSpecPix))


def test_prior():

    num_components = 3
    numObj = 10
    prior = PriorModel(num_components)

    params = prior.random(key)
    assert_shape(params, prior.params_shape)

    redshifts = 10 ** jax.random.normal(key, (numObj,))
    mu = prior.get_mean_at_z(params, redshifts)
    assert_shape(mu, (numObj, num_components))
    muloginvvar = prior.get_loginvvar_at_z(params, redshifts)
    assert_shape(muloginvvar, (numObj, num_components))


def test_bayesianpca_spec_and_specandphot():

    numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ = 122, 100, 47, 5, 50
    datapipeline = DataPipeline.save_fake_data(
        numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ
    )
    datapipeline = DataPipeline("data/fake/fake_")

    numComponents, numPoly = 4, 3
    params_list, pcacomponents_prior = init_params(
        key, numObj, numComponents, numPoly, numSedPix
    )

    batch_size = 20
    indices = datapipeline.ind_train_local
    data_batch = datapipeline.next_batch(indices, batch_size)
    polynomials_spec = chebychevPolynomials(numPoly, numSpecPix)
    aux_data = (polynomials_spec, numSpecPix, 0)

    result = bayesianpca_spec_and_specandphot(params_list, data_batch, aux_data)

    for x in result:
        assert np.all((np.isfinite(x)))

    (
        logfml_speconly,
        theta_map_speconly,
        theta_std_speconly,
        specmod_map_speconly,
        logfml_specandphot,
        theta_map_specandphot,
        theta_std_specandphot,
        specmod_map_specandphot,
    ) = result
    assert_shape(theta_map_speconly, (batch_size, numComponents + numPoly))
    assert_shape(theta_std_speconly, (batch_size, numComponents + numPoly))
    assert_shape(specmod_map_speconly, (batch_size, numSpecPix))
    assert_shape(theta_map_specandphot, (batch_size, numComponents + numPoly))
    assert_shape(theta_std_specandphot, (batch_size, numComponents + numPoly))
    assert_shape(specmod_map_specandphot, (batch_size, numSpecPix))

    loss = loss_spec_and_specandphot(params_list, data_batch, aux_data)
    assert np.all(np.isfinite(loss))
