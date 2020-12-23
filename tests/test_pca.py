from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *
import itertools
import jax.experimental.optimizers

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
    dataPipeline = DataPipeline.save_fake_data(
        numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    numComponents, numPoly = 4, 3
    params_list, pcacomponents_prior = init_params(
        key, numObj, numComponents, numPoly, numSedPix
    )

    batch_size = 20
    indices = dataPipeline.ind_train_local
    data_batch = dataPipeline.next_batch(indices, batch_size)
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


def test_loss_spec_and_specandphot():

    numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    numComponents, numPoly = 4, 3
    params_list, pcacomponents_prior = init_params(
        key, numObj, numComponents, numPoly, numSedPix
    )

    batch_size = 20
    polynomials_spec = chebychevPolynomials(numPoly, numSpecPix)
    aux_data = (polynomials_spec, numSpecPix, 0)

    data_batch = dataPipeline.next_batch(dataPipeline.ind_train_local, batch_size)
    loss = loss_spec_and_specandphot(params_list, data_batch, aux_data)
    assert np.all(np.isfinite(loss))

    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(1e-3)
    opt_state = opt_init(params_list)

    @partial(jit, static_argnums=(2, 3))
    def update(step, opt_state, data_batch, data_aux):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_spec_and_specandphot)(
            params, data_batch, data_aux
        )
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    nbatches = dataPipeline.get_nbatches(dataPipeline.ind_train_local, batch_size)
    n_epoch = 4
    itercount = itertools.count()
    for i in range(n_epoch):
        neworder = jax.random.permutation(key, dataPipeline.ind_train_local.size)
        train_indices_reordered = np.take(dataPipeline.ind_train_local, neworder)
        dataPipeline.batch = 0  # reset batch number
        for j in range(nbatches):
            data_batch = dataPipeline.next_batch(train_indices_reordered, batch_size)
            loss, opt_state = update(next(itercount), opt_state, data_batch, aux_data)
            assert np.all(np.isfinite(loss))
