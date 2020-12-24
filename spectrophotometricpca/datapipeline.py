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
    mask[x == 0] = val
    mask[x_var <= 0] = val
    mask[x_var == mask_std_val ** 2.0] = val
    fullindices = indices[:, None] + onp.arange(x.shape[1])[None, :]
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

    def load_spectrophotometry(self, root, write_subset=False, use_subset=False):

        self.root = root

        self.lamgrid = np.load(self.root + "lamgrid.npy")
        self.lam_phot_eff = np.load(self.root + "lam_phot_eff.npy")
        self.lam_phot_size_eff = np.load(self.root + "lam_phot_size_eff.npy")
        self.transferfunctions = np.load(self.root + "transferfunctions.npy")
        self.transferfunctions_zgrid = np.load(
            self.root + "transferfunctions_zgrid.npy"
        )

        assert self.transferfunctions.shape[0] == self.transferfunctions_zgrid.size
        assert self.transferfunctions.shape[1] == self.lamgrid.size

        if use_subset:
            suffix = "2.npy"
        else:
            suffix = ".npy"

        self.chi2s_sdss = np.load(self.root + "chi2s_sdss" + suffix)
        self.lamspec_waveoffset = int(
            np.load(self.root + "lamspec_waveoffset" + suffix)
        )
        self.index_wave = np.load(self.root + "index_wave" + suffix)
        self.index_transfer_redshift = np.load(
            self.root + "index_transfer_redshift" + suffix
        )
        self.spec = np.load(self.root + "spec" + suffix)
        self.specmod_sdss = np.load(self.root + "spec_mod" + suffix)
        self.spec_invvar = np.load(self.root + "spec_invvar" + suffix)
        self.phot = fluxes = np.load(self.root + "phot" + suffix)
        self.phot_invvar = flux_ivars = np.load(self.root + "phot_invvar" + suffix)
        self.redshifts = np.load(self.root + "redshifts" + suffix)

        numObj = self.chi2s_sdss.shape[0]
        assert_shape(self.chi2s_sdss, (numObj,))
        assert_shape(self.index_wave, (numObj,))
        assert_shape(self.index_transfer_redshift, (numObj,))
        numSpecPix = self.spec.shape[1]
        assert_shape(self.spec, (numObj, numSpecPix))
        assert_shape(self.specmod_sdss, (numObj, numSpecPix))
        assert_shape(self.spec_invvar, (numObj, numSpecPix))
        numPhotBands = self.phot.shape[1]
        assert_shape(self.phot, (numObj, numPhotBands))
        assert_shape(self.phot_invvar, (numObj, numPhotBands))
        assert_shape(self.redshifts, (numObj,))

        self.numObj, self.numPhotBands = self.phot.shape
        self.numSpecPixels = self.spec.shape[1]

        if write_subset:

            M = 50000
            suffix = "2.npy"

            self.index_wave = self.index_wave[:M]
            self.redshifts = self.redshifts[:M]
            self.chi2s_sdss = self.chi2s_sdss[:M]
            self.phot_invvar = self.phot_invvar[:M, :]
            self.index_transfer_redshift = self.index_transfer_redshift[:M]

            np.save(self.root + "index_wave" + suffix, self.index_wave[:M])
            np.save(
                self.root + "index_transfer_redshift2.npy",
                self.index_transfer_redshift,
            )
            np.save(self.root + "redshifts" + suffix, self.redshifts)
            np.save(self.root + "spec" + suffix, self.spec)
            np.save(self.root + "chi2s_sdss" + suffix, self.chi2s_sdss)
            np.save(self.root + "spec_invvar" + suffix, self.spec_invvar)
            np.save(self.root + "phot" + suffix, self.phot)
            np.save(self.root + "phot_invvar" + suffix, self.phot_invvar)
            np.save(self.root + "spec_mod" + suffix, self.specmod_sdss)

    @staticmethod
    def save_fake_data(numObj, numSedPix, numSpecPix, numPhotBands, numTransferZ):

        root = "data/fake/fake_"
        from jax.random import uniform, randint

        np.save(root + "lamgrid.npy", 8.1e2 + np.arange(numSedPix))
        np.save(root + "lam_phot_eff.npy", np.arange(numPhotBands))
        np.save(root + "lam_phot_size_eff.npy", np.arange(numPhotBands))
        np.save(
            root + "transferfunctions.npy",
            uniform(key, (numTransferZ, numSedPix, numPhotBands)),
        )
        np.save(root + "transferfunctions_zgrid.npy", np.arange(numTransferZ))

        np.save(root + "chi2s_sdss.npy", uniform(key, (numObj,)))
        np.save(
            root + "lamspec_waveoffset.npy",
            randint(key, (1,), 0, numSedPix - numSpecPix - 1),
        )
        np.save(
            root + "index_wave.npy",
            randint(key, (numObj,), 0, numSedPix - numSpecPix - 1),
        )
        np.save(
            root + "index_transfer_redshift.npy",
            randint(key, (numObj,), 0, numTransferZ),
        )
        np.save(root + "spec.npy", uniform(key, (numObj, numSpecPix)))
        np.save(root + "spec_mod.npy", uniform(key, (numObj, numSpecPix)))
        np.save(root + "spec_invvar.npy", uniform(key, (numObj, numSpecPix)))
        np.save(root + "phot.npy", uniform(key, (numObj, numPhotBands)))
        np.save(root + "phot_invvar.npy", uniform(key, (numObj, numPhotBands)))
        np.save(root + "redshifts.npy", uniform(key, (numObj,)))

    def __init__(
        self,
        root,
        subSampling=1,
        npix_min=1,
        lambda_start=8e2,
        write_subset=False,
        use_subset=False,
        selected_bands=None,
    ):

        self.load_spectrophotometry(
            root, write_subset=write_subset, use_subset=use_subset
        )

        self.batch = 0
        if selected_bands is None:
            self.selected_bands = np.arange(self.numPhotBands)
        else:
            self.selected_bands = selected_bands

        if subSampling > 1:

            numIR = 0  # 200
            self.lamgrid = np.concatenate(
                (self.lamgrid[:-numIR][::subSampling], self.lamgrid[-numIR:])
            )
            self.transferfunctions = np.hstack(
                (
                    self.transferfunctions[:, :-numIR, :][:, ::subSampling, :],
                    self.transferfunctions[:, -numIR:, :],
                )
            )
            self.transferfunctions = self.transferfunctions[::subSampling, :, :]
            self.transferfunctions_zgrid = self.transferfunctions_zgrid[::subSampling]
            self.lamspec_waveoffset = self.lamspec_waveoffset // subSampling
            # self.initial_pca = np.hstack(
            #    (self.initial_pca[:, :-numIR][:, ::subSampling], self.initial_pca[:, -numIR:])
            # )

        xbounds = onp.zeros(self.lamgrid.size + 1)
        xbounds[1:-1] = (self.lamgrid[1:] + self.lamgrid[:-1]) / 2
        xbounds[0] = self.lamgrid[0] - (xbounds[1] - self.lamgrid[0])
        xbounds[-1] = self.lamgrid[-1] + (self.lamgrid[-1] - xbounds[-2])
        xsizes = np.asarray(np.diff(xbounds))
        # Multiplying by delta lambda in preparation for integral over lambda
        self.transferfunctions[:, :, :] *= xsizes[None, :, None]

        # Truncate a small section to reduce memory requirements.
        idx_start = onp.where(self.lamgrid > lambda_start)[0][0]
        print("idx_start", idx_start)
        self.lamgrid = np.asarray(self.lamgrid[idx_start:])
        self.transferfunctions = np.asarray(self.transferfunctions[:, idx_start:, :])
        # self.initial_pca = self.initial_pca[:, idx_start:]
        self.index_wave = np.asarray(self.index_wave - idx_start)
        self.lamspec_waveoffset = np.asarray(self.lamspec_waveoffset - idx_start)

        print("Initial lamgrid shape:", self.lamgrid.shape)
        print("Initial spec shape:", self.spec.shape)
        print("Initial phot shape:", self.phot.shape)
        # spec[spec <= 0] = np.nan
        self.spec_invvar[self.spec_invvar <= 0] = 0

        if subSampling > 1:
            self.spec = self.spec[:, ::subSampling]
            self.specmod_sdss = self.specmod_sdss[:, ::subSampling]
            self.spec_invvar = self.spec_invvar[:, ::subSampling]
            self.index_wave = self.index_wave // subSampling
            self.index_transfer_redshift = self.index_transfer_redshift // subSampling

        ind = ~np.isfinite(self.spec)
        ind |= ~np.isfinite(self.spec_invvar)
        # ind |= spec <= 0
        ind |= self.spec_invvar <= 0
        ind = np.where(ind)[0]
        self.spec[ind] = 0.0
        self.spec_invvar[ind] = 0.0

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

        # renormalize all data
        ind = self.spec != 0
        ind |= self.spec_invvar != 0
        ind = np.where(ind)[0]
        self.mean_spec, self.std_spec = (
            np.mean(self.spec[ind]),
            np.std(self.spec[ind]),
        )
        self.mean_phot, self.std_phot = np.mean(self.phot), np.std(self.phot)

        # Floor photometric errors
        ind = self.phot_invvar ** -0.5 < 1e-2 * self.phot
        ind = np.where(ind)[0]
        self.phot_invvar[ind] = (1e-2 * self.phot[ind]) ** -2.0

        print("Finished pre-processing data.")
        print("Revised data shape:", self.spec.shape)

        n_spec = self.spec.shape[0]
        self.size = self.spec.shape[1]

        # masks = create_mask(self.spec, self.spec_invvar, self.index_wave)
        masks = ~(self.spec_invvar == 0)
        npix = np.sum(masks, axis=1)
        print("Number of objects with 0 valid pixels:", np.sum(npix == 0))
        print("Number of objects with <10 valid pixels:", np.sum(npix <= 10))
        print("Number of objects with <100 valid pixels:", np.sum(npix <= 100))
        ind_ok = onp.where(npix > npix_min)[0]
        print("Number of objects with valid pixels:", ind_ok.size)

        if True:
            n_valid = min([50000, n_spec // 2])
            onp.random.shuffle(ind_ok)
            self.ind_train_orig = ind_ok[n_valid:-1]
            self.ind_valid_orig = ind_ok[0:n_valid]
        else:
            # Uniform validation
            n_valid = min([80000, n_spec // 2])
            self.ind_valid_orig = ind_ok[draw_uniform(redshifts[ind_ok], 30, n_valid)]
            self.ind_train_orig = ind_ok
            self.ind_train_orig = self.ind_train_orig[
                ~np.in1d(self.ind_train_orig, self.ind_valid_orig, assume_unique=True)
            ]
            self.ind_train_orig = self.ind_train_orig[
                draw_uniform(
                    redshifts[self.ind_train_orig],
                    30,
                    120000 - self.ind_valid_orig.size // 2,
                )
            ]
            self.ind_train_orig = np.concatenate(
                [self.ind_train_orig, self.ind_valid_orig[::2]]
            )
            self.ind_valid_orig = self.ind_valid_orig[1::2]

        onp.random.shuffle(self.ind_train_orig)
        onp.random.shuffle(self.ind_valid_orig)

        print(n_valid, ind_ok.size)
        print("Size of training:", self.ind_train_orig.size)
        print("Size of validation:", self.ind_valid_orig.size)

        ind_sel = np.concatenate([1 * self.ind_train_orig, 1 * self.ind_valid_orig])
        self.ind_train_local = np.arange(self.ind_train_orig.size)
        self.ind_valid_local = np.arange(
            self.ind_train_orig.size,
            self.ind_train_orig.size + self.ind_valid_orig.size,
        )

        # chi2s_sdss = np.sum(masks*(specmod_sdss - spec)**2*spec_invvar, axis=-1)
        # np.save(self.root+'/chi2s_sdss', chi2s_sdss)
        # stop

        self.index_wave = self.index_wave[ind_sel]
        self.index_transfer_redshift = self.index_transfer_redshift[ind_sel]
        self.spec = self.spec[ind_sel, :]
        self.chi2s_sdss = self.chi2s_sdss[ind_sel]
        # self.specmod_sdss = specmod_sdss[ind_sel, :]
        self.spec_invvar = self.spec_invvar[ind_sel, :]
        self.phot = self.phot[ind_sel, :]
        self.phot_invvar = self.phot_invvar[ind_sel, :]
        self.redshifts = self.redshifts[ind_sel]
        self.specphotscalings = np.ones_like(self.redshifts)

    def get_grids(self):
        numSedPix = self.lamgrid.size
        numSpecPix = self.spec.shape[1]
        numBands = self.phot.shape[1]
        lamgrid_spec = lamgrid[
            self.lamspec_waveoffset : self.lamspec_waveoffset + numSpecPix
        ]
        return (
            self.lamgrid,
            self.lam_phot_eff,
            self.lam_phot_size_eff,
            self.transferfunctions,
            self.transferfunctions_zgrid,
            self.selected_bands,
            numSedPix,
            numSpecPix,
            numBands,
            lamgrid_spec,
        )

    def next_batch(self, indices, batch_size, random_masking=False):
        length = indices.size
        startindex = self.batch * batch_size
        batch_indices = indices[startindex : startindex + batch_size]
        batch_index_wave = np.take(self.index_wave, batch_indices)
        batch_index_transfer_redshift = np.take(
            self.index_transfer_redshift, batch_indices
        )
        batch_spec = np.take(self.spec, batch_indices, axis=0)
        batch_spec_invvar = np.take(self.spec_invvar, batch_indices, axis=0)
        # batch_sed_mask = create_mask(batch_spec, batch_spec_invvar, batch_index_wave)
        batch_phot = np.take(self.phot, batch_indices, axis=0)
        batch_phot_invvar = np.take(self.phot_invvar, batch_indices, axis=0)
        batch_redshifts = np.take(self.redshifts, batch_indices)
        batch_specphotscaling = np.take(self.specphotscalings, batch_indices)
        self.batch += 1

        nextbatch_startindex = self.batch * batch_size
        if nextbatch_startindex >= length:
            self.batch = 0

        if random_masking:  # Random masking of pixels
            heights = jax.random.uniform(
                0, 1, batch_spec.shape[0] * batch_spec.shape[1]
            ).reshape(batch_spec.shape)
            thresholds_cut = jax.random.uniform(0.0, 1.0, batch_spec.shape[0])[
                :, None
            ] * np.ones_like(batch_spec)
            batch_sed_mask[heights < thresholds_cut] = 0

        # batch_spec_invvar[batch_sed_mask == 0] = 0

        actualbatchsize = min([batch_size, length - startindex])

        # si, _ = startindex, batch_size
        # sh = (batch_spec.shape[0], wavesize)
        # batch_transferfunctions = np.zeros((batch_spec.shape[0], wavesize, batch_phot.shape[1]))
        batch_transferfunctions = self.transferfunctions[
            batch_index_transfer_redshift, :, :
        ]
        batch_index_wave_ext = batch_index_wave[:, None] + np.arange(
            batch_spec.shape[1]
        )
        print(
            "Number of negative wave indices:",
            np.sum(batch_index_wave < 0),
            np.sum(batch_index_wave_ext < 0),
        )
        if np.sum(batch_index_wave_ext < 0) > 0:
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

    def get_nbatches(self, indices, batch_size):
        self.batch_size = batch_size
        return (indices.shape[0] // self.batch_size) + 1
