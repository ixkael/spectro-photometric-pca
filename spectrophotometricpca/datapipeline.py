import numpy as onp
import jax
import jax.numpy as np
import random
import os
import scipy.interpolate
import astropy.io.fits as pyfits

from chex import assert_shape

key = jax.random.PRNGKey(42)

mask_std_val = 1e2


def process_time(start_time, end_time, multiply=1, no_hours=False):
    dt = (end_time - start_time) * multiply
    if no_hours:
        hour = 0
    else:
        hour = int(dt / 3600)
    min = int((dt - 3600 * hour) / 60)
    sec = int(dt - 3600 * hour - 60 * min)
    if no_hours:
        return (min, sec)
    else:
        return (hour, min, sec)


def create_mask(x, x_var, indices, val=0.0):
    mask = onp.ones(x.shape)
    # mask[x == 0] = val
    ind = np.where(np.logical_or(x_var <= 0, x_var == mask_std_val ** 2.0))
    mask[ind] = val
    fullindices = onp.asarray(indices)[:, None] + onp.arange(x.shape[1])[None, :]
    mask[fullindices < 0] = val
    # offs = - np.maximum(indices, np.zeros_like(indices))
    # for io, off in enumerate(offs):
    #    mask[io, 0:off] = 0
    return np.asarray(mask)


def interp(x_new, x, y):
    return scipy.interpolate.interp1d(
        x, y, fill_value="extrapolate", kind="linear", bounds_error=False
    )(x_new)


def draw_uniform(samples, bins, desired_size):
    """
    Draw uniform set of samples


    """
    hist, bin_edges = np.histogram(samples, bins=bins)
    avg_nb = int(desired_size / float(bins))
    numbers = np.repeat(avg_nb, bins)
    for j in range(4):
        numbers[hist <= numbers] = hist[hist <= numbers]
        nb_rest = desired_size - np.sum(numbers[hist <= numbers])  # * bins
        avg_nb = round(nb_rest / np.sum(hist > numbers))
        numbers[hist > numbers] = avg_nb

    result = []
    count = 0
    for i in range(bin_edges.size - 1):
        ind = samples >= bin_edges[i]
        ind &= samples <= bin_edges[i + 1]
        if ind.sum() > 0:
            positions = np.where(ind)[0]
            nb = min([numbers[i], ind.sum()])
            result.append(jax.random.choice(positions, nb, replace=False))

    return np.concatenate(result)


class DataPipeline:
    """
    Pipeline for loading data
    """

    def load_spectrophotometry(
        self,
        input_dir="./",
        write_subset=False,
        use_subset=False,
        subsampling=1,
    ):

        self.input_dir = input_dir

        self.lamgrid = onp.load(self.input_dir + "lamgrid.npy")
        self.lam_phot_eff = onp.load(self.input_dir + "lam_phot_eff.npy")
        self.lam_phot_size_eff = onp.load(self.input_dir + "lam_phot_size_eff.npy")
        self.transferfunctions = onp.load(self.input_dir + "transferfunctions.npy")
        self.transferfunctions_zgrid = onp.load(
            self.input_dir + "transferfunctions_zgrid.npy"
        )

        assert self.transferfunctions.shape[0] == self.transferfunctions_zgrid.size
        assert self.transferfunctions.shape[1] == self.lamgrid.size

        if use_subset:
            suffix = "2.npy"
        else:
            suffix = ".npy"

        self.chi2s_sdss = onp.load(self.input_dir + "chi2s_sdss" + suffix)
        self.lamspec_waveoffset = int(
            onp.load(self.input_dir + "lamspec_waveoffset" + suffix)
        )
        self.index_wave = onp.load(self.input_dir + "index_wave" + suffix)
        self.index_transfer_redshift = onp.load(
            self.input_dir + "index_transfer_redshift" + suffix
        )
        self.spec = onp.load(self.input_dir + "spec" + suffix)
        self.specmod_sdss = onp.load(self.input_dir + "spec_mod" + suffix)
        self.spec_invvar = onp.load(self.input_dir + "spec_invvar" + suffix)
        self.phot = fluxes = onp.load(self.input_dir + "phot" + suffix)
        self.phot_invvar = flux_ivars = onp.load(
            self.input_dir + "phot_invvar" + suffix
        )
        self.redshifts = onp.load(self.input_dir + "redshifts" + suffix)

        n_obj = self.chi2s_sdss.shape[0]
        assert_shape(self.chi2s_sdss, (n_obj,))
        assert_shape(self.index_wave, (n_obj,))
        assert_shape(self.index_transfer_redshift, (n_obj,))
        n_pix_spec = self.spec.shape[1]
        assert_shape(self.spec, (n_obj, n_pix_spec))
        assert_shape(self.specmod_sdss, (n_obj, n_pix_spec))
        assert_shape(self.spec_invvar, (n_obj, n_pix_spec))
        n_pix_phot = self.phot.shape[1]
        assert_shape(self.phot, (n_obj, n_pix_phot))
        assert_shape(self.phot_invvar, (n_obj, n_pix_phot))
        assert_shape(self.redshifts, (n_obj,))

        self.n_obj, self.n_pix_phot = self.phot.shape
        self.n_pix_specels = self.spec.shape[1]

        if write_subset:

            M = 50000
            suffix = "2.npy"

            self.index_wave = self.index_wave[:M]
            self.redshifts = self.redshifts[:M]
            self.chi2s_sdss = self.chi2s_sdss[:M]
            self.phot_invvar = self.phot_invvar[:M, :]
            self.index_transfer_redshift = self.index_transfer_redshift[:M]

            np.save(self.input_dir + "index_wave" + suffix, self.index_wave[:M])
            np.save(
                self.input_dir + "index_transfer_redshift2.npy",
                self.index_transfer_redshift,
            )
            np.save(self.input_dir + "redshifts" + suffix, self.redshifts)
            np.save(self.input_dir + "spec" + suffix, self.spec)
            np.save(self.input_dir + "chi2s_sdss" + suffix, self.chi2s_sdss)
            np.save(self.input_dir + "spec_invvar" + suffix, self.spec_invvar)
            np.save(self.input_dir + "phot" + suffix, self.phot)
            np.save(self.input_dir + "phot_invvar" + suffix, self.phot_invvar)
            np.save(self.input_dir + "spec_mod" + suffix, self.specmod_sdss)

        if subsampling > 1:

            self.lamgrid = self.lamgrid[::subsampling]
            self.transferfunctions = self.transferfunctions[:, ::subsampling, :][
                ::subsampling, :, :
            ]
            self.transferfunctions_zgrid = self.transferfunctions_zgrid[::subsampling]
            self.lamspec_waveoffset = self.lamspec_waveoffset // subsampling
            self.spec = self.spec[:, ::subsampling]
            self.specmod_sdss = self.specmod_sdss[:, ::subsampling]
            self.spec_invvar = self.spec_invvar[:, ::subsampling]
            self.index_wave = self.index_wave // subsampling
            self.index_transfer_redshift = self.index_transfer_redshift // subsampling

    @staticmethod
    def save_fake_data(n_obj, n_pix_sed, n_pix_spec, n_pix_phot, n_pix_transfer):

        root = "data/fake/fake_"
        from jax.random import uniform, randint

        np.save(root + "lamgrid.npy", 8.1e2 + np.arange(n_pix_sed))
        np.save(root + "lam_phot_eff.npy", np.arange(n_pix_phot))
        np.save(root + "lam_phot_size_eff.npy", np.arange(n_pix_phot))
        np.save(
            root + "transferfunctions.npy",
            uniform(key, (n_pix_transfer, n_pix_sed, n_pix_phot)),
        )
        np.save(root + "transferfunctions_zgrid.npy", np.arange(n_pix_transfer))

        np.save(root + "chi2s_sdss.npy", uniform(key, (n_obj,)))
        np.save(
            root + "lamspec_waveoffset.npy",
            randint(key, (1,), 0, n_pix_sed - n_pix_spec - 1),
        )
        np.save(
            root + "index_wave.npy",
            randint(key, (n_obj,), 0, n_pix_sed - n_pix_spec - 1),
        )
        np.save(
            root + "index_transfer_redshift.npy",
            randint(key, (n_obj,), 0, n_pix_transfer),
        )
        np.save(root + "spec.npy", uniform(key, (n_obj, n_pix_spec)))
        np.save(root + "spec_mod.npy", uniform(key, (n_obj, n_pix_spec)))
        np.save(root + "spec_invvar.npy", uniform(key, (n_obj, n_pix_spec)))
        np.save(root + "phot.npy", uniform(key, (n_obj, n_pix_phot)))
        np.save(root + "phot_invvar.npy", uniform(key, (n_obj, n_pix_phot)))
        np.save(root + "redshifts.npy", uniform(key, (n_obj,)))

    def __init__(
        self,
        input_dir="./",
        subsampling=1,
        npix_min=1,
        lambda_start=8e2,
        write_subset=False,
        use_subset=False,
    ):

        self.load_spectrophotometry(
            input_dir=input_dir,
            write_subset=write_subset,
            use_subset=use_subset,
            subsampling=subsampling,
        )

        self.batch = 0
        self.subsampling = subsampling
        self.npix_min = npix_min
        self.lambda_start = lambda_start

        # Multiplying by delta lambda in preparation for integral over lambda
        xbounds = onp.zeros(self.lamgrid.size + 1)
        xbounds[1:-1] = (self.lamgrid[1:] + self.lamgrid[:-1]) / 2
        xbounds[0] = self.lamgrid[0] - (xbounds[1] - self.lamgrid[0])
        xbounds[-1] = self.lamgrid[-1] + (self.lamgrid[-1] - xbounds[-2])
        xsizes = np.asarray(np.diff(xbounds))
        self.transferfunctions = self.transferfunctions * xsizes[None, :, None]

        if lambda_start is not None:
            # Truncate a small section to reduce memory requirements.
            idx_start = onp.where(self.lamgrid > lambda_start)[0][0]
            print("idx_start", idx_start)
            self.lamgrid = np.asarray(self.lamgrid[idx_start:])
            self.transferfunctions = np.asarray(
                self.transferfunctions[:, idx_start:, :]
            )
            # self.initial_pca = self.initial_pca[:, idx_start:]
            self.index_wave = np.asarray(self.index_wave - idx_start)
            self.lamspec_waveoffset = np.asarray(self.lamspec_waveoffset - idx_start)

        print("Initial lamgrid shape:", self.lamgrid.shape)
        print("Initial spec shape:", self.spec.shape)
        print("Initial phot shape:", self.phot.shape)
        # spec[spec <= 0] = np.nan
        self.spec_invvar[~onp.isfinite(self.spec)] = 0
        self.spec_invvar[~onp.isfinite(self.spec_invvar)] = 0
        self.spec_invvar[self.spec_invvar < 0] = 0
        self.spec[~onp.isfinite(self.spec)] = 0

        # Masking sky lines
        lamsize_spec = self.spec.shape[1]
        lamgrid_spec = self.lamgrid[
            np.arange(self.lamspec_waveoffset, self.lamspec_waveoffset + lamsize_spec)
        ]
        ind = np.logical_and(lamgrid_spec >= 6860, lamgrid_spec <= 6920)
        ind |= np.logical_and(lamgrid_spec >= 7150, lamgrid_spec <= 7340)
        ind |= np.logical_and(lamgrid_spec >= 7575, lamgrid_spec <= 7725)
        print(
            "Number of pixels masked due to skylines:", np.sum(ind), "out of", ind.size
        )
        ind = np.where(ind)[0]
        self.spec[:, ind] = 0.0
        self.spec_invvar[:, ind] = 0.0

        # Floor spectroscopic errors
        ind = self.spec_invvar ** -0.5 < 1e-3 * np.abs(self.spec)
        ind = np.where(ind)[0]
        print("How many spec errors are floored?", np.sum(ind), "out of", ind.size)
        self.spec_invvar[ind] = (1e-3 * np.abs(self.spec)[ind]) ** -2.0

        # Calculated after changing the data
        self.chi2s_sdss = np.sum(
            (self.specmod_sdss - self.spec) ** 2 * self.spec_invvar, axis=-1
        )

        # Floor photometric errors
        ind = self.phot_invvar ** -0.5 < 1e-2 * self.phot
        ind = np.where(ind)[0]
        self.phot_invvar[ind] = (1e-2 * self.phot[ind]) ** -2.0

        self.specphotscalings = np.ones((self.spec.shape[0],))

        print("Finished pre-processing data.")
        print("Revised data shape:", self.spec.shape)

        # masks = create_mask(self.spec, self.spec_invvar, self.index_wave)
        masks = ~(self.spec_invvar == 0)
        npix = np.sum(masks, axis=1)
        print("Number of objects with 0 valid pixels:", np.sum(npix == 0))
        print("Number of objects with <10 valid pixels:", np.sum(npix <= 10))
        print("Number of objects with <100 valid pixels:", np.sum(npix <= 100))
        self.indices = onp.where(npix > npix_min)[0]
        onp.random.shuffle(self.indices)
        print("Number of objects with valid pixels:", self.indices.size)

    def get_grids(self):
        n_pix_sed = self.lamgrid.size
        n_pix_spec = self.spec.shape[1]
        n_pix_phot = self.phot.shape[1]
        lamgrid_spec = self.lamgrid[
            self.lamspec_waveoffset : self.lamspec_waveoffset + n_pix_spec
        ]
        return (
            self.lamgrid,
            self.lam_phot_eff,
            self.lam_phot_size_eff,
            self.transferfunctions,
            self.transferfunctions_zgrid,
            n_pix_sed,
            n_pix_spec,
            n_pix_phot,
            lamgrid_spec,
        )

    def next_batch(self, indices, batchsize):
        length = indices.size
        startindex = self.batch * batchsize
        batch_indices = indices[startindex : startindex + batchsize]
        batch_index_wave = np.take(self.index_wave, batch_indices)
        batch_index_transfer_redshift = np.take(
            self.index_transfer_redshift, batch_indices
        )
        batch_spec = np.take(self.spec, batch_indices, axis=0)
        batch_spec_invvar = np.take(self.spec_invvar, batch_indices, axis=0)
        batch_sed_mask = create_mask(batch_spec, batch_spec_invvar, batch_index_wave)
        batch_phot = np.take(self.phot, batch_indices, axis=0)
        batch_phot_invvar = np.take(self.phot_invvar, batch_indices, axis=0)
        batch_redshifts = np.take(self.redshifts, batch_indices)
        batch_specphotscaling = np.take(self.specphotscalings, batch_indices)
        self.batch += 1

        nextbatch_startindex = self.batch * batchsize
        if nextbatch_startindex >= length:
            self.batch = 0

        actualbatchsize = min([batchsize, length - startindex])

        # si, _ = startindex, batchsize
        # sh = (batch_spec.shape[0], wavesize)
        # batch_transferfunctions = np.zeros((batch_spec.shape[0], wavesize, batch_phot.shape[1]))
        batch_transferfunctions = self.transferfunctions[
            batch_index_transfer_redshift, :, :
        ]
        batch_index_wave_ext = batch_index_wave[:, None] + np.arange(
            batch_spec.shape[1]
        )
        if np.sum(batch_index_wave_ext < 0) > 0:
            print(
                "Number of negative wave indices:",
                np.sum(batch_index_wave < 0),
                np.sum(batch_index_wave_ext < 0),
            )
            exit(1)
        # batch_index_wave_ext[batch_index_wave_ext < 0] = 0

        batch_spec_loginvvar = np.where(
            batch_spec_invvar == 0, 0, np.log(batch_spec_invvar)
        )
        batch_phot_loginvvar = np.where(
            batch_phot_invvar == 0, 0, np.log(batch_phot_invvar)
        )

        return (
            startindex,
            actualbatchsize,
            batch_index_wave,
            batch_index_transfer_redshift,
            batch_spec,
            batch_spec_invvar,
            batch_spec_loginvvar,
            # batch_sed_mask,
            batch_specphotscaling,
            batch_phot,
            batch_phot_invvar,
            batch_phot_loginvvar,
            batch_redshifts,
            batch_transferfunctions,
            batch_index_wave_ext,
        )

    def get_nbatches(self, indices, batchsize):
        self.batchsize = batchsize
        return (indices.shape[0] // self.batchsize) + 1


class ResultsPipeline:
    def __init__(self, prefix, suffix, n_components, dataPipeline, indices=None):
        n_pix_sed = dataPipeline.lamgrid.size
        n_pix_spec = dataPipeline.spec.shape[1]
        n_pix_phot = dataPipeline.phot.shape[1]
        self.dataPipeline = dataPipeline

        self.prefix = prefix
        self.suffix = suffix

        if indices is not None:
            n_obj = indices.size
            self.indices = indices
            self.logfml = onp.zeros((n_obj,))
            self.specmod = onp.zeros((n_obj, n_pix_spec))
            self.photmod = onp.zeros((n_obj, n_pix_phot))
            self.thetamap = onp.zeros((n_obj, n_components))
            self.thetastd = onp.zeros((n_obj, n_components))

    def write_batch(self, data_batch, logfml, thetamap, thetastd, specmod, photmod):

        si, bs = data_batch[0], data_batch[1]
        self.logfml[si : si + bs] = logfml
        self.specmod[si : si + bs, :] = specmod
        self.photmod[si : si + bs, :] = photmod
        self.thetamap[si : si + bs, :] = thetamap
        self.thetastd[si : si + bs, :] = thetastd

    def load_reconstructions(self):

        self.indices = oonp.load(self.prefix + "indices" + self.suffix + ".npy")
        self.logfml = oonp.load(self.prefix + "logfml" + self.suffix + ".npy")
        self.specmod = oonp.load(self.prefix + "specmod" + self.suffix + ".npy")
        self.photmod = oonp.load(self.prefix + "photmod" + self.suffix + ".npy")
        self.thetamap = oonp.load(self.prefix + "thetamap" + self.suffix + ".npy")
        self.thetastd = oonp.load(self.prefix + "thetastd" + self.suffix + ".npy")

    def write_reconstructions(self):

        onp.save(self.prefix + "indices" + self.suffix, self.indices)
        onp.save(self.prefix + "logfml" + self.suffix, self.logfml)
        onp.save(self.prefix + "specmod" + self.suffix, self.specmod)
        onp.save(self.prefix + "photmod" + self.suffix, self.photmod)
        onp.save(self.prefix + "thetamap" + self.suffix, self.thetamap)
        onp.save(self.prefix + "thetastd" + self.suffix, self.thetastd)


def extract_pca_parameters(runroot):

    last = runroot.split("/")[-1]
    vals = onp.array(last.split("_")[1::2])
    print(vals)
    n_components, n_poly, batchsize, subsampling = vals[[0, 1, 2, 3]].astype(int)
    learningrate = vals[-1].astype(float)

    return n_components, n_poly, batchsize, subsampling, learningrate


def pca_file_prefix(n_components, n_poly, batchsize, subsampling, learningrate):

    prefix = "pca_"
    prefix += str(n_components) + "_components_"
    prefix += str(n_poly) + "_poly_"
    prefix += str(batchsize) + "_batchsize_"
    prefix += str(subsampling) + "_subsampling_"
    prefix += str(learningrate) + "_learningrate"

    return prefix
