from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_bayesianpca_spec_and_specandphot():

    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 122, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    dataPipeline = DataPipeline("data/fake/fake_")

    batchsize = 20
    data_batch = dataPipeline.next_batch(dataPipeline.indices, batchsize)

    (
        si,
        bs,
        batch_index_wave,
        batch_index_transfer_redshift,
        spec,
        spec_invvar,
        spec_loginvvar,
        # batch_spec_mask,
        specphotscaling,
        phot,
        phot_invvar,
        phot_loginvvar,
        batch_redshifts,
        batch_transferfunctions,
        batch_index_wave_ext,
        batch_interprightindices,
        batch_interpweights,
    ) = data_batch

    assert bs == batchsize
    assert si == 0
    assert_shape(spec, (bs, n_pix_spec))
    assert_shape(spec_invvar, (bs, n_pix_spec))
    assert_shape(spec_loginvvar, (bs, n_pix_spec))
    assert_shape(phot, (bs, n_pix_phot))
    assert_shape(phot_invvar, (bs, n_pix_phot))
    assert_shape(phot_loginvvar, (bs, n_pix_phot))
    assert_shape(batch_redshifts, (bs,))
    assert_shape(batch_transferfunctions, (bs, n_pix_sed, n_pix_phot))

    n_components = 3
    pcacomponents_speconly = jax.random.normal(key, (n_components, n_pix_sed))
    pcacomponents_speconly_atz = take_batch(
        pcacomponents_speconly, batch_index_wave, n_pix_spec
    )
    assert_shape(pcacomponents_speconly_atz, (bs, n_components, n_pix_spec))


def test_results():

    subsampling = 1
    n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer = 22, 100, 47, 5, 50
    dataPipeline = DataPipeline.save_fake_data(
        n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer
    )
    input_dir = "data/fake/fake_"
    dataPipeline = DataPipeline(input_dir=input_dir, subsampling=subsampling)

    prefix, suffix = "data/fake/fakeprefix_", "_fakesuffix"
    n_components = 4
    batchsize = 10
    indices = np.arange(n_obj // 4, n_obj // 2)
    n_obj_out = indices.size

    resultsPipeline = ResultsPipeline(
        prefix, suffix, n_components, dataPipeline, indices
    )

    n_batches = dataPipeline.get_nbatches(indices, batchsize)

    logfml = jax.random.normal(key, (n_obj_out,))
    thetamap = jax.random.normal(key, (n_obj_out, n_components))
    thetastd = jax.random.normal(key, (n_obj_out, n_components))
    specmod = jax.random.normal(key, (n_obj_out, n_pix_spec))
    photmod = jax.random.normal(key, (n_obj_out, n_pix_phot))
    ellfactors = jax.random.normal(key, (n_obj_out,))

    for _ in range(n_batches):
        data_batch = dataPipeline.next_batch(indices, batchsize)

        si, bs = data_batch[0], data_batch[1]
        print(si, si + bs)
        resultsPipeline.write_batch(
            data_batch,
            logfml[si : si + bs],
            thetastd[si : si + bs, :],
            thetastd[si : si + bs, :],
            specmod[si : si + bs, :],
            photmod[si : si + bs, :],
            ellfactors[si : si + bs],
        )

    np.allclose(logfml, resultsPipeline.logfml)
    np.allclose(specmod, resultsPipeline.specmod)
    np.allclose(photmod, resultsPipeline.photmod)
    np.allclose(thetamap, resultsPipeline.thetamap)
    np.allclose(thetastd, resultsPipeline.thetastd)
    np.allclose(ellfactors, resultsPipeline.ellfactors)

    resultsPipeline.write_reconstructions()

    resultsPipeline.load_reconstructions()

    np.allclose(logfml, resultsPipeline.logfml)
    np.allclose(specmod, resultsPipeline.specmod)
    np.allclose(photmod, resultsPipeline.photmod)
    np.allclose(thetamap, resultsPipeline.thetamap)
    np.allclose(thetastd, resultsPipeline.thetastd)
    np.allclose(ellfactors, resultsPipeline.ellfactors)

    resultsPipeline2 = ResultsPipeline(
        prefix, suffix, n_components, dataPipeline, indices
    )
    resultsPipeline2.load_reconstructions()

    np.allclose(indices, resultsPipeline2.indices)
    np.allclose(logfml, resultsPipeline2.logfml)
    np.allclose(specmod, resultsPipeline2.specmod)
    np.allclose(photmod, resultsPipeline2.photmod)
    np.allclose(thetamap, resultsPipeline2.thetamap)
    np.allclose(thetastd, resultsPipeline2.thetastd)
    np.allclose(ellfactors, resultsPipeline2.ellfactors)


def test_filennames():
    opt_basis, opt_priors = 0, 1
    n_components, n_poly, batchsize, subsampling, learningrate = 3, 3, 13, 1, 1e-3
    prefix = pca_file_prefix(
        n_components,
        n_poly,
        batchsize,
        subsampling,
        opt_basis,
        opt_priors,
        learningrate,
    )
    (
        n_components2,
        n_poly2,
        batchsize2,
        subsampling2,
        opt_basis2,
        opt_priors2,
        learningrate2,
    ) = extract_pca_parameters(prefix)
    assert n_components == n_components2
    assert n_poly == n_poly2
    assert learningrate == learningrate2
    assert batchsize == batchsize2
    assert subsampling == subsampling2
    assert opt_basis == opt_basis2
    assert opt_priors == opt_priors2


def test_load_fits_templates():

    log_lam1 = np.arange(3.55, 4.02, 0.0001)
    log_lam2 = np.arange(2.85, 3.55, 0.0001)
    log_lam = np.concatenate([log_lam1, log_lam2])
    lam = 10.0 ** log_lam

    num_components = 4

    y_new = load_fits_templates(lam, num_components)

    assert y_new.shape[0] == num_components
    assert y_new.shape[1] == lam.size


def test_interp():

    nobj = 3
    n_components = 2
    n_pix_sed = 4
    n_pix_spec = 5

    x_grid = onp.linspace(0, 1, n_pix_sed)
    x_target = onp.random.uniform(0, 1, n_pix_spec * nobj).reshape((nobj, n_pix_spec))
    y_grid = onp.cos(x_grid * 10)[None, :] * onp.ones((n_components, n_pix_sed))

    # interprightindices = np.random.randint(1, n_pix_sed-1, n_pix_spec*nobj).reshape((nobj, n_pix_spec))
    # interpweights = np.random.randn(nobj, n_pix_spec)
    interprightindices, interpweights = interp_coefficients(x_grid, x_target)

    transfer = onp.zeros((nobj, n_pix_spec, n_pix_sed))
    for io in range(nobj):
        for ipix in range(n_pix_spec):
            transfer[io, ipix, interprightindices[io, ipix] - 1] = interpweights[
                io, ipix
            ]
            transfer[io, ipix, interprightindices[io, ipix]] = (
                1 - interpweights[io, ipix]
            )

    transfer2 = create_interp_transfer(interprightindices, interpweights, n_pix_sed)

    assert np.allclose(transfer, transfer2)

    # Need to transpose to have suitable shapez
    y_target = np.transpose(
        np.dot(transfer, y_grid.T), [0, 2, 1]
    )  # nobj, ncomp, n_pix_spec
    y_target2 = onp.zeros_like(y_target)
    for io in range(nobj):
        for ic in range(n_components):
            y_target2[io, ic, :] = scipy.interpolate.interp1d(
                x_grid[:],
                y_grid[ic, :],
                kind="linear",
                bounds_error=True,
                fill_value=0,
                assume_sorted=True,
            )(x_target[io, :])
    print(y_target.shape, y_target2.shape)
    assert np.allclose(y_target, y_target2)
