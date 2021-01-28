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
    # assert_shape(params, prior.params_shape)

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

    prefix, suffix = "data/fake/fake_", ""
    n_components, n_poly = 4, 3
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)
    pcamodel = PCAModel(polynomials_spec, prefix, suffix)
    pcacomponents_prior_speconly = pcamodel.init_params(
        key, n_components, n_poly, n_pix_sed
    )
    params_speconly = pcamodel.get_params()
    pcacomponents_prior_specandphot = pcamodel.init_params(
        key, n_components, n_poly, n_pix_sed
    )
    params_specandphot = pcamodel.get_params()
    print("pcamodel.get_params()", len(pcamodel.get_params()))
    pcamodel.set_params(params_specandphot)

    batchsize = 20
    indices = dataPipeline.indices
    data_batch = dataPipeline.next_batch(indices, batchsize)

    result_speconly = pcamodel.bayesianpca_speconly(
        params_speconly, data_batch, polynomials_spec
    )
    result_specandphot = pcamodel.bayesianpca_specandphot(
        params_specandphot, data_batch, polynomials_spec
    )

    for x in result_speconly + result_specandphot:
        assert np.all((np.isfinite(x)))

    for result in [result_speconly, result_specandphot]:
        (logfml, thetamap, thetastd, specmod_map, photmod_map, ellfactors) = result
        assert_shape(thetamap, (batchsize, n_components + n_poly))
        assert_shape(thetastd, (batchsize, n_components + n_poly))
        assert_shape(specmod_map, (batchsize, n_pix_spec))
        assert_shape(photmod_map, (batchsize, n_pix_phot))
        assert_shape(ellfactors, (batchsize,))

    params_all = [params_speconly, params_specandphot]

    @partial(jit, static_argnums=())
    def loss_spec_and_specandphot(params_all, data_batch, polynomials_spec):
        [params_speconly, params_specandphot] = params_all
        return pcamodel.loss_speconly(
            params_speconly, data_batch, polynomials_spec
        ) + pcamodel.loss_specandphot(params_specandphot, data_batch, polynomials_spec)

    loss_value = loss_spec_and_specandphot(params_all, data_batch, polynomials_spec)
    assert np.all(np.isfinite(loss_value))

    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(1e-3)
    opt_state = opt_init(params_all)

    @partial(jit, static_argnums=())
    def update(step, opt_state, data_batch, data_aux):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_spec_and_specandphot)(
            params, data_batch, data_aux
        )
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    nbatches = dataPipeline.get_nbatches(dataPipeline.indices, batchsize)
    n_epoch = 2
    itercount = itertools.count()
    for i in range(n_epoch):
        neworder = jax.random.permutation(key, dataPipeline.indices.size)
        train_indices_reordered = np.take(dataPipeline.indices, neworder)
        dataPipeline.batch = 0  # reset batch number
        for j in range(nbatches):
            data_batch = dataPipeline.next_batch(train_indices_reordered, batchsize)
            loss, opt_state = update(
                next(itercount), opt_state, data_batch, polynomials_spec
            )
            assert np.all(np.isfinite(loss))


def test_batch_indices():

    n_obj = 3
    n_components = 2
    n_pix_spec = 10
    n_pix_sed = 100

    pcacomponents_speconly = jax.random.normal(key, (n_components, n_pix_sed))

    batch_index_wave = jax.random.randint(key, (n_obj,), 0, n_pix_sed - n_pix_spec)

    indices_0, indices_1 = batch_indices(batch_index_wave, n_components, n_pix_spec)

    pcacomponents_speconly_atz = pcacomponents_speconly[indices_0, indices_1]

    assert_shape(pcacomponents_speconly_atz, (n_obj, n_components, n_pix_spec))

    for i in range(n_obj):
        off = batch_index_wave[i]
        assert np.all(
            pcacomponents_speconly_atz[i, :, :]
            == pcacomponents_speconly[:, off : off + n_pix_spec]
        )
