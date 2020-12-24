from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *
import itertools
import jax.experimental.optimizers

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_chebychevPolynomials():
    n_poly, n_pix_spec = 3, 100
    res = chebychevPolynomials(n_poly, n_pix_spec)
    assert_shape(res, (n_poly, n_pix_spec))


def test_prior():

    n_components = 3
    n_obj = 10
    prior = PriorModel(n_components)

    params = prior.random(key)
    assert_shape(params, prior.params_shape)

    redshifts = 10 ** jax.random.normal(key, (n_obj,))
    mu = prior.get_mean_at_z(params, redshifts)
    assert_shape(mu, (n_obj, n_components))
    muloginvvar = prior.get_loginvvar_at_z(params, redshifts)
    assert_shape(muloginvvar, (n_obj, n_components))


def test_bayesianpca_spec_and_specandphot():

    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    n_components, n_poly = 4, 3
    params_list, pcacomponents_prior = init_params(
        key, n_obj, n_components, n_poly, n_pix_sed
    )

    batchsize = 20
    indices = dataPipeline.ind_train_local
    data_batch = dataPipeline.next_batch(indices, batchsize)
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)
    aux_data = (polynomials_spec, n_pix_spec, 0)

    result = bayesianpca_spec_and_specandphot(params_list, data_batch, aux_data)

    for x in result:
        assert np.all((np.isfinite(x)))

    (
        logfml_speconly,
        theta_map_speconly,
        theta_std_speconly,
        specmod_map_speconly,
        photmod_map_speconly,
        logfml_specandphot,
        theta_map_specandphot,
        theta_std_specandphot,
        specmod_map_specandphot,
        photmod_map_specandphot,
    ) = result
    assert_shape(theta_map_speconly, (batchsize, n_components + n_poly))
    assert_shape(theta_std_speconly, (batchsize, n_components + n_poly))
    assert_shape(specmod_map_speconly, (batchsize, n_pix_spec))
    assert_shape(photmod_map_speconly, (batchsize, n_pix_phot))
    assert_shape(theta_map_specandphot, (batchsize, n_components + n_poly))
    assert_shape(theta_std_specandphot, (batchsize, n_components + n_poly))
    assert_shape(specmod_map_specandphot, (batchsize, n_pix_spec))
    assert_shape(photmod_map_specandphot, (batchsize, n_pix_phot))


def test_loss_spec_and_specandphot():

    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    n_components, n_poly = 4, 3
    params_list, pcacomponents_prior = init_params(
        key, n_obj, n_components, n_poly, n_pix_sed
    )

    batchsize = 20
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)
    aux_data = (polynomials_spec, n_pix_spec, 0)

    data_batch = dataPipeline.next_batch(dataPipeline.ind_train_local, batchsize)
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

    nbatches = dataPipeline.get_nbatches(dataPipeline.ind_train_local, batchsize)
    n_epoch = 2
    itercount = itertools.count()
    for i in range(n_epoch):
        neworder = jax.random.permutation(key, dataPipeline.ind_train_local.size)
        train_indices_reordered = np.take(dataPipeline.ind_train_local, neworder)
        dataPipeline.batch = 0  # reset batch number
        for j in range(nbatches):
            data_batch = dataPipeline.next_batch(train_indices_reordered, batchsize)
            loss, opt_state = update(next(itercount), opt_state, data_batch, aux_data)
            assert np.all(np.isfinite(loss))
