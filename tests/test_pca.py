from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *
import itertools
import jax.experimental.optimizers

import pytest
import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


@pytest.mark.skip()
def test_bayesianpca_photonly():

    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    dataPipeline = DataPipeline("data/fake/fake_", phot=True, spec=False)

    prefix, suffix = "data/fake/fake_", ""
    n_components, n_poly = 4, 3
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)
    pcamodel = PCAModel(polynomials_spec, prefix, suffix)

    regularization = 1e-3

    for opt_basis in [False, True]:
        for opt_priors in [False, True]:
            if not opt_priors and not opt_basis:
                continue

            print("Opt_basis:", opt_basis)
            print("Opt prior:", opt_priors)

            bayesianpca = jit(bayesianpca_photonly, static_argnums=(3, 4, 5))

            @partial(jit, static_argnums=(3, 4, 5, 6))
            def loss_fn(
                params,
                data_batch,
                data_aux,
                n_components,
                n_pix_spec,
                opt_basis,
                opt_priors,
                regularization,
            ):
                if opt_basis and opt_priors:
                    pcacomponents = params[0]
                    pcacomponents_init = data_aux[1]
                if opt_basis and not opt_priors:
                    pcacomponents = params[0]
                    pcacomponents_init = data_aux[2]
                if not opt_basis and opt_priors:
                    pcacomponents = data_aux[0]
                    pcacomponents_init = data_aux[2]
                # pcacomponents -= pcacomponents.mean(axis=1)[:, None]
                (logfml, _, _, _, _, _) = bayesianpca(
                    params,
                    data_batch,
                    data_aux,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                )
                diff = pcacomponents - pcacomponents_init
                return -np.sum(logfml)  # + np.sum(diff ** 2) * regularization

            pcacomponents_prior = pcamodel.init_params(
                key,
                n_components,
                n_poly,
                n_pix_sed,
                opt_basis,
                opt_priors,
            )
            pcacomponents_init = 1 * pcamodel.pcacomponents
            params = pcamodel.get_params_opt()
            pcamodel.set_params(params)

            batchsize = 20
            indices = dataPipeline.indices
            data_batch = dataPipeline.next_batch_photonly(indices, batchsize)

            if opt_basis and opt_priors:
                data_aux = pcacomponents_init
                components = params[0]
                priors = params[1]
            if opt_basis and not opt_priors:
                components = params[0]
                priors = pcamodel.get_params_nonopt()[0]
                data_aux = (priors, pcacomponents_init)
            if not opt_basis and opt_priors:
                components = pcamodel.get_params_nonopt()[0]
                data_aux = (components, pcacomponents_init)
                priors = params
            print("components.shape", components.shape)

            result = bayesianpca(
                params,
                data_batch,
                data_aux,
                n_components,
                opt_basis,
                opt_priors,
            )

            lossval = loss_fn(
                params,
                data_batch,
                data_aux,
                n_components,
                opt_basis,
                opt_priors,
                regularization,
            )
            assert np.all((np.isfinite(lossval)))

            (
                logfml,
                thetamap,
                thetastd,
                photmod_map,
            ) = result
            assert_shape(thetamap, (batchsize, n_components))
            assert_shape(thetastd, (batchsize, n_components))
            assert_shape(photmod_map, (batchsize, n_pix_phot))

            opt_init, opt_update, get_params_opt = jax.experimental.optimizers.adam(
                1e-3
            )
            opt_state = opt_init(params)

            @partial(jit, static_argnums=(4, 5, 6, 7, 8))
            def update(
                step,
                opt_state,
                data_batch,
                data_aux,
                n_components,
                opt_basis,
                opt_priors,
                regularization,
            ):
                params = get_params_opt(opt_state)
                value, grads = jax.value_and_grad(loss_fn)(
                    params,
                    data_batch,
                    data_aux,
                    n_components,
                    opt_basis,
                    opt_priors,
                    regularization,
                )
                opt_state = opt_update(step, grads, opt_state)
                return value, opt_state

            nbatches = dataPipeline.get_nbatches(dataPipeline.indices, batchsize)
            n_epoch = 3
            itercount = itertools.count()
            for i in range(n_epoch):
                neworder = jax.random.permutation(key, dataPipeline.indices.size)
                train_indices_reordered = np.take(dataPipeline.indices, neworder)
                dataPipeline.batch = 0  # reset batch number
                for j in range(nbatches):
                    data_batch = dataPipeline.next_batch_photonly(
                        train_indices_reordered, batchsize
                    )
                    loss, opt_state = update(
                        next(itercount),
                        opt_state,
                        data_batch,
                        data_aux,
                        n_components,
                        opt_basis,
                        opt_priors,
                        regularization,
                    )
                    assert np.all(np.isfinite(loss.block_until_ready()))


def test_chebychevPolynomials():
    n_poly, n_pix_spec = 3, 100
    res = chebychevPolynomials(n_poly, n_pix_spec)
    assert_shape(res, (n_poly, n_pix_spec))


def test_prior():

    n_archetypes = 2
    n_components = 3
    n_obj = 10
    prior = PriorModel(n_archetypes, n_components)

    params = prior.random(key)
    # assert_shape(params, prior.params_shape)

    redshifts = 10 ** jax.random.normal(key, (n_obj,))
    mu = prior.get_mean_at_z(params, redshifts)
    assert_shape(mu, (n_obj, n_archetypes, n_components))
    muloginvvar = prior.get_loginvvar_at_z(params, redshifts)
    assert_shape(muloginvvar, (n_obj, n_archetypes, n_components))


def test_bayesianpca_spec_and_specandphot():

    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    prefix, suffix = "data/fake/fake_", ""
    n_archetypes, n_components, n_poly = 2, 4, 3
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)
    pcamodel = PCAModel(polynomials_spec, prefix, suffix)

    regularization = 1e-3

    for opt_basis in [False, True]:
        for opt_priors in [False, True]:
            if not opt_priors and not opt_basis:
                continue

            print("Opt_basis:", opt_basis)
            print("Opt prior:", opt_priors)
            for speconly in [True, False]:

                print("speconly", speconly)

                if speconly:
                    bayesianpca = jit(
                        bayesianpca_speconly, static_argnums=(3, 4, 5, 6, 7)
                    )
                else:
                    bayesianpca = jit(
                        bayesianpca_specandphot, static_argnums=(3, 4, 5, 6, 7)
                    )

                from jax.scipy.special import logsumexp

                @partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
                def loss_fn(
                    params,
                    data_batch,
                    data_aux,
                    n_archetypes,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                    regularization,
                ):
                    if opt_basis and opt_priors:
                        pcacomponents = params[0]
                        pcacomponents_init = data_aux[1]
                    if opt_basis and not opt_priors:
                        pcacomponents = params[0]
                        pcacomponents_init = data_aux[2]
                    if not opt_basis and opt_priors:
                        pcacomponents = data_aux[0]
                        pcacomponents_init = data_aux[2]
                    # pcacomponents -= pcacomponents.mean(axis=1)[:, None]
                    (logfml, _, _, _, _, _) = bayesianpca(
                        params,
                        data_batch,
                        data_aux,
                        n_archetypes,
                        n_components,
                        n_pix_spec,
                        opt_basis,
                        opt_priors,
                    )
                    diff = pcacomponents - pcacomponents_init
                    return -np.sum(
                        logsumexp(logfml, axis=1)
                    )  # + np.sum(diff ** 2) * regularization

                pcacomponents_prior = pcamodel.init_params(
                    key,
                    n_archetypes,
                    n_components,
                    n_poly,
                    n_pix_sed,
                    opt_basis,
                    opt_priors,
                )
                pcacomponents_init = 1 * pcamodel.pcacomponents
                params = pcamodel.get_params_opt()
                print("params", params)
                pcamodel.set_params(params)

                batchsize = 20
                indices = dataPipeline.indices
                data_batch = dataPipeline.next_batch_specandphot(indices, batchsize)

                if opt_basis and opt_priors:
                    data_aux = (polynomials_spec, pcacomponents_init)
                    components = params[0]
                    priors = [params[1], params[2], params[3]]
                if opt_basis and not opt_priors:
                    components = params[0]
                    priors = pcamodel.get_params_nonopt()
                    data_aux = (priors, polynomials_spec, pcacomponents_init)
                if not opt_basis and opt_priors:
                    components = pcamodel.get_params_nonopt()[0]
                    data_aux = (components, polynomials_spec, pcacomponents_init)
                    priors = params

                result = bayesianpca(
                    params,
                    data_batch,
                    data_aux,
                    n_archetypes,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                )

                lossval = loss_fn(
                    params,
                    data_batch,
                    data_aux,
                    n_archetypes,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                    regularization,
                )
                assert np.all((np.isfinite(lossval)))

                (
                    logfml,
                    thetamap,
                    thetastd,
                    specmod_map,
                    photmod_map,
                    ellfactors,
                ) = result
                assert_shape(thetamap, (batchsize, n_archetypes, n_components + n_poly))
                assert_shape(thetastd, (batchsize, n_archetypes, n_components + n_poly))
                assert_shape(specmod_map, (batchsize, n_archetypes, n_pix_spec))
                assert_shape(photmod_map, (batchsize, n_archetypes, n_pix_phot))
                assert_shape(ellfactors, (batchsize, n_archetypes))

                opt_init, opt_update, get_params_opt = jax.experimental.optimizers.adam(
                    1e-3
                )
                opt_state = opt_init(params)

                @partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
                def update(
                    step,
                    opt_state,
                    data_batch,
                    data_aux,
                    n_archetypes,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                    regularization,
                ):
                    params = get_params_opt(opt_state)
                    value, grads = jax.value_and_grad(loss_fn)(
                        params,
                        data_batch,
                        data_aux,
                        n_archetypes,
                        n_components,
                        n_pix_spec,
                        opt_basis,
                        opt_priors,
                        regularization,
                    )
                    opt_state = opt_update(step, grads, opt_state)
                    return value, opt_state

                nbatches = dataPipeline.get_nbatches(dataPipeline.indices, batchsize)
                n_epoch = 3
                itercount = itertools.count()
                for i in range(n_epoch):
                    neworder = jax.random.permutation(key, dataPipeline.indices.size)
                    train_indices_reordered = np.take(dataPipeline.indices, neworder)
                    dataPipeline.batch = 0  # reset batch number
                    for j in range(nbatches):
                        data_batch = dataPipeline.next_batch_specandphot(
                            train_indices_reordered, batchsize
                        )
                        loss, opt_state = update(
                            next(itercount),
                            opt_state,
                            data_batch,
                            data_aux,
                            n_archetypes,
                            n_components,
                            n_pix_spec,
                            opt_basis,
                            opt_priors,
                            regularization,
                        )
                        assert np.all(np.isfinite(loss.block_until_ready()))


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
        assert np.allclose(
            pcacomponents_speconly_atz[i, :, :],
            pcacomponents_speconly[:, off : off + n_pix_spec],
        )
