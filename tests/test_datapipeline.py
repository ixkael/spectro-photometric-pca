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
    sedmod = jax.random.normal(key, (n_obj_out, n_pix_sed))

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
            sedmod[si : si + bs, :],
        )

    np.allclose(logfml, resultsPipeline.logfml)
    np.allclose(specmod, resultsPipeline.specmod)
    np.allclose(photmod, resultsPipeline.photmod)
    np.allclose(thetamap, resultsPipeline.thetamap)
    np.allclose(thetastd, resultsPipeline.thetastd)
    np.allclose(sedmod, resultsPipeline.sedmod)

    resultsPipeline.write_reconstructions()

    resultsPipeline.load_reconstructions()

    np.allclose(logfml, resultsPipeline.logfml)
    np.allclose(specmod, resultsPipeline.specmod)
    np.allclose(photmod, resultsPipeline.photmod)
    np.allclose(thetamap, resultsPipeline.thetamap)
    np.allclose(thetastd, resultsPipeline.thetastd)
    np.allclose(sedmod, resultsPipeline.sedmod)

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
    np.allclose(sedmod, resultsPipeline2.sedmod)


def test_filennames():
    n_components, learningrate, batchsize, subsampling = 3, 1e-3, 13, 1
    prefix = pca_file_prefix(n_components, learningrate, batchsize, subsampling)
    n_components2, learningrate2, batchsize2, subsampling2 = extract_pca_parameters(
        prefix
    )
    assert n_components == n_components2
    assert learningrate == learningrate2
    assert batchsize == batchsize2
    assert subsampling == subsampling2
