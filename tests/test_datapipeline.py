from spectrophotometricpca.pca import *
from spectrophotometricpca.datapipeline import *

import jax
import jax.numpy as np

from chex import assert_shape

key = jax.random.PRNGKey(42)


def test_bayesianpca_spec_and_specandphot():

    numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ = 122, 100, 47, 5, 50
    datapipeline = DataPipeline.save_fake_data(
        numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ
    )
    datapipeline = DataPipeline("data/fake/fake_", npix_min=1)

    batch_size = 20
    data_batch = datapipeline.next_batch(datapipeline.ind_train_local, batch_size)

    print(datapipeline.ind_train_local)
    (
        si,
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
        bs,
        batch_transferfunctions,
        batch_index_wave_ext,
    ) = data_batch

    assert bs == batch_size
    assert si == 0
    assert_shape(spec, (bs, numSpecPix))
    assert_shape(spec_invvar, (bs, numSpecPix))
    assert_shape(spec_loginvvar, (bs, numSpecPix))
    assert_shape(phot, (bs, numPhotBands))
    assert_shape(phot_invvar, (bs, numPhotBands))
    assert_shape(phot_loginvvar, (bs, numPhotBands))
    assert_shape(batch_redshifts, (bs,))
    assert_shape(batch_transferfunctions, (bs, numSedPix, numPhotBands))
