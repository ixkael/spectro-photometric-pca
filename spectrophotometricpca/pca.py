# -*- coding: utf-8 -*-

import numpy as onp

import jax
from jax import partial, jit
import jax.numpy as np

from gasp.pca_utils_jx import *
from gasp.marginallikelihoods_jx import *


def chebychevPolynomials(n_poly, n_pix_spec):
    x = np.linspace(-1.0, 1.0, n_pix_spec)
    res = np.vstack(
        [x * 0 + 1, x, 2 * x ** 2 - 1, 4 * x ** 3 - 3 * x, 8 * x ** 4 - 8 * x ** 2 + 1]
    )  # (n_poly, n_pix_spec)
    print(res.shape)
    return res[0:n_poly, :]


class PriorModel:
    """"""

    def __init__(self, n_components):
        self.n_components = n_components
        self.params_shape = (self.n_components, 2)

    def random(self, key):
        params = jax.random.normal(key, self.params_shape)
        return params

    @staticmethod
    def get_mean_at_z(params, redshifts):
        return np.ones((redshifts.size, 1)) * params[None, :, 0]

    @staticmethod
    def get_loginvvar_at_z(params, redshifts):
        return np.ones((redshifts.size, 1)) * params[None, :, 1]


@jit
def bayesianpca_speconly(
    components_spec,  # [n_obj, n_components, nspec]
    polynomials_spec,  # [n_poly, nspec]
    spec,  # [n_obj, nspec]
    spec_invvar,  # [n_obj, nspec]
    spec_loginvvar,  # [n_obj, nspec]
    components_prior_mean,  # [n_obj, n_components]
    components_prior_loginvvar,  # [n_obj, n_components]
    polynomials_prior_mean,  # [n_poly]
    polynomials_prior_loginvvar,  # [n_poly]
):

    n_obj, nspec = np.shape(spec)[0], np.shape(spec)[1]
    n_components = np.shape(components_spec)[1]
    n_poly = np.shape(polynomials_spec)[0]

    components_spec_all = np.concatenate(
        [components_spec, polynomials_spec[None, :, :] * np.ones((n_obj, 1, nspec))],
        axis=-2,
    )  # [n_obj, n_components+n_poly, nspec]

    mu = np.concatenate(
        [components_prior_mean, polynomials_prior_mean[None, :] * np.ones((n_obj, 1))],
        axis=-1,
    )
    logmuinvvar = np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, :] * np.ones((n_obj, 1)),
        ],
        axis=-1,
    )
    muinvvar = np.exp(logmuinvvar)

    logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        thetamap,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_twotransfers_jitvmap(
        components_spec_all,
        spec,
        spec_invvar,
        spec_loginvvar,
        mu,
        muinvvar,
        logmuinvvar,
    )

    thetastd = np.diagonal(theta_cov, axis1=1, axis2=2) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all[:, :, :] * thetamap[:, :, None], axis=1)
    # chi2_spec = np.sum((specmod_map - spec) ** 2 * spec_invvar, axis=-1)

    return (logfml, thetamap, thetastd, specmod_map)


@jit
def bayesianpca_specandphot(
    components_spec,  # [n_obj, n_components, nspec]
    components_phot,  # [n_obj, n_components, nphot]
    polynomials_spec,  # [n_poly, nspec]
    ellfactors,  # [n_obj, ]
    spec,  # [n_obj, nspec]
    spec_invvar,  # [n_obj, nspec]
    spec_loginvvar,  # [n_obj, nspec]
    phot,  # [n_obj, nphot]
    phot_invvar,  # [n_obj, nphot]
    phot_loginvvar,  # [n_obj, nphot]
    components_prior_mean,  # [n_obj, n_components]
    components_prior_loginvvar,  # [n_obj, n_components]
    polynomials_prior_mean,  # [n_poly]
    polynomials_prior_loginvvar,  # [n_poly]
):

    n_obj, nspec = np.shape(spec)[0], np.shape(spec)[1]
    n_components = np.shape(components_spec)[1]
    n_poly = np.shape(polynomials_spec)[0]
    nphot = np.shape(phot)[1]

    components_spec_all = np.concatenate(
        [components_spec, polynomials_spec[None, :, :] * np.ones((n_obj, 1, nspec))],
        axis=-2,
    )  # [n_obj, n_components+n_poly, nspec]

    zeros = np.zeros((n_obj, n_poly, nphot))
    components_phot_all = np.concatenate(
        [components_phot, zeros], axis=1
    )  # [n_obj, n_components+n_poly, nphot]

    mu = np.concatenate(
        [components_prior_mean, polynomials_prior_mean[None, :] * np.ones((n_obj, 1))],
        axis=-1,
    )
    logmuinvvar = np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, :] * np.ones((n_obj, 1)),
        ],
        axis=-1,
    )
    muinvvar = np.exp(logmuinvvar)

    logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        thetamap,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_threetransfers_jitvmap(
        ellfactors,
        components_spec_all,
        components_phot_all,
        spec,
        spec_invvar,
        spec_loginvvar,
        phot,
        phot_invvar,
        phot_loginvvar,
        mu,
        muinvvar,
        logmuinvvar,
    )

    thetastd = np.diagonal(theta_cov, axis1=1, axis2=2) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all[:, :, :] * thetamap[:, :, None], axis=1)
    photmod_map = np.sum(components_phot_all[:, :, :] * thetamap[:, :, None], axis=1)
    # chi2_spec = np.sum((specmod_map - spec) ** 2 * spec_invvar, axis=-1)
    # chi2_phot = np.sum((photmod_map - phot) ** 2 * phot_invvar, axis=-1)

    return (logfml, thetamap, thetastd, specmod_map, photmod_map)


@partial(jit, static_argnums=(1, 2))
def batch_indices(start_indices, n_components, npix):
    nobj = start_indices.shape[0]

    indices_2d = start_indices[:, None] + np.arange(npix)[None, :]

    indices_0 = np.arange(n_components)[None, :, None] * np.ones(
        (nobj, n_components, npix), dtype=int
    )
    indices_1 = indices_2d[:, None, :] * np.ones((1, n_components, npix), dtype=int)

    return indices_0, indices_1


class PCAModel:
    def __init__(self, polynomials_spec, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix
        self.polynomials_spec = polynomials_spec

    @partial(jit, static_argnums=(0))
    def loss_speconly(
        self,
        params,
        data_batch,
        polynomials_spec,
    ):
        (logfml_speconly, _, _, _, _) = self.bayesianpca_speconly(
            params,
            data_batch,
            polynomials_spec,
        )
        return -np.sum(logfml_speconly)

    @partial(jit, static_argnums=(0))
    def loss_specandphot(
        self,
        params,
        data_batch,
        polynomials_spec,
    ):
        (logfml_specandphot, _, _, _, _) = self.bayesianpca_specandphot(
            params,
            data_batch,
            polynomials_spec,
        )
        return -np.sum(logfml_specandphot)

    def write_model(self):

        np.save(self.prefix + "pcacomponents" + self.suffix, self.pcacomponents)
        np.save(
            self.prefix + "components_prior_params" + self.suffix,
            self.components_prior_params,
        )
        np.save(
            self.prefix + "polynomials_prior_mean" + self.suffix,
            self.polynomials_prior_mean,
        )
        np.save(
            self.prefix + "polynomials_prior_loginvvar" + self.suffix,
            self.polynomials_prior_loginvvar,
        )

    def load_model(self):

        self.pcacomponents = np.load(self.prefix + "pcacomponents" + self.suffix)
        self.components_prior_params = np.load(
            self.prefix + "components_prior_params" + self.suffix + ".npy",
        )
        self.polynomials_prior_mean = np.load(
            self.prefix + "polynomials_prior_mean" + self.suffix + ".npy",
        )
        self.polynomials_prior_loginvvar = np.load(
            self.prefix + "polynomials_prior_loginvvar" + self.suffix + ".npy",
        )

    def init_params(self, key, n_components, n_poly, lamgridsize):

        self.pcacomponents_prior = PriorModel(n_components)

        self.pcacomponents = jax.random.normal(key, (n_components, lamgridsize))
        self.components_prior_params = self.pcacomponents_prior.random(key)
        self.polynomials_prior_mean = jax.random.normal(key, (n_poly,))
        self.polynomials_prior_loginvvar = jax.random.normal(key, (n_poly,))

        return self.pcacomponents_prior

    def get_params(self):
        arr = [
            self.pcacomponents,
            self.components_prior_params,
            self.polynomials_prior_mean,
            self.polynomials_prior_loginvvar,
        ]
        return arr

    def set_params(self, params):
        self.pcacomponents = params[0]
        self.components_prior_params = params[1]
        self.polynomials_prior_mean = params[2]
        self.polynomials_prior_loginvvar = params[3]

    @partial(jit, static_argnums=(0))
    def bayesianpca_speconly(
        self,
        params,
        data_batch,
        polynomials_spec,
    ):
        (
            pcacomponents_speconly,
            components_prior_params_speconly,
            polynomials_prior_mean_speconly,
            polynomials_prior_loginvvar_speconly,
        ) = params

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

        # ones = np.ones((bs, 1))
        # batch_index_wave = np.zeros((bs,), int)

        n_components = pcacomponents_speconly.shape[0]
        n_pix_spec = spec.shape[1]
        indices_0, indices_1 = batch_indices(batch_index_wave, n_components, n_pix_spec)

        pcacomponents_speconly_atz = pcacomponents_speconly[indices_0, indices_1]
        components_phot_speconly = np.sum(
            pcacomponents_speconly[None, :, :, None]
            * batch_transferfunctions[:, None, :, :],
            axis=2,
        )
        components_prior_mean_speconly = PriorModel.get_mean_at_z(
            components_prior_params_speconly, batch_redshifts
        )
        components_prior_loginvvar_speconly = PriorModel.get_loginvvar_at_z(
            components_prior_params_speconly, batch_redshifts
        )

        (
            logfml_speconly,
            thetamap_speconly,
            thetastd_speconly,
            specmod_map_speconly,
        ) = bayesianpca_speconly(
            pcacomponents_speconly_atz,  # [n_obj, n_components, nspec]
            polynomials_spec,  # [n_poly, nspec]
            spec,  # [n_obj, nspec]
            spec_invvar,  # [n_obj, nspec]
            spec_loginvvar,  # [n_obj, nspec]
            components_prior_mean_speconly,  # [n_obj, n_components]
            components_prior_loginvvar_speconly,  # [n_obj, n_components]
            polynomials_prior_mean_speconly,
            polynomials_prior_loginvvar_speconly,
        )
        photmod_map_speconly = np.sum(
            components_phot_speconly[:, :, :]
            * thetamap_speconly[:, 0:n_components, None],
            axis=1,
        )
        return (
            logfml_speconly,
            thetamap_speconly,
            thetastd_speconly,
            specmod_map_speconly,
            photmod_map_speconly,
        )

    @partial(jit, static_argnums=(0))
    def bayesianpca_specandphot(
        self,
        params,
        data_batch,
        polynomials_spec,
    ):
        (
            pcacomponents_specandphot,
            components_prior_params_specandphot,
            polynomials_prior_mean_specandphot,
            polynomials_prior_loginvvar_specandphot,
        ) = params

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

        ellfactors = np.ones_like(batch_redshifts)
        # ones = np.ones((bs, 1))

        n_components = pcacomponents_specandphot.shape[0]
        n_pix_spec = spec.shape[1]
        indices_0, indices_1 = batch_indices(batch_index_wave, n_components, n_pix_spec)
        pcacomponents_specandphot_atz = pcacomponents_specandphot[indices_0, indices_1]
        components_phot_specandphot = np.sum(
            pcacomponents_specandphot[None, :, :, None]
            * batch_transferfunctions[:, None, :, :],
            axis=2,
        )
        components_prior_mean_specandphot = PriorModel.get_mean_at_z(
            components_prior_params_specandphot, batch_redshifts
        )
        components_prior_loginvvar_specandphot = PriorModel.get_loginvvar_at_z(
            components_prior_params_specandphot, batch_redshifts
        )

        (
            logfml_specandphot,
            thetamap_specandphot,
            thetastd_specandphot,
            specmod_map_specandphot,
            photmod_map_specandphot,
        ) = bayesianpca_specandphot(
            pcacomponents_specandphot_atz,  # [n_obj, n_components, nspec]
            components_phot_specandphot,  # [n_obj, n_components, nphot]
            polynomials_spec,  # [n_poly, nspec]
            ellfactors,  # [n_obj, ]
            spec,  # [n_obj, nspec]
            spec_invvar,  # [n_obj, nspec]
            spec_loginvvar,  # [n_obj, nspec]
            phot,  # [n_obj, nphot]
            phot_invvar,  # [n_obj, nphot]
            phot_loginvvar,  # [n_obj, nphot]
            components_prior_mean_specandphot,  # [n_obj, n_components]
            components_prior_loginvvar_specandphot,  # [n_obj, n_components]
            polynomials_prior_mean_specandphot,
            polynomials_prior_loginvvar_specandphot,
        )
        return (
            logfml_specandphot,
            thetamap_specandphot,
            thetastd_specandphot,
            specmod_map_specandphot,
            photmod_map_specandphot,
        )
