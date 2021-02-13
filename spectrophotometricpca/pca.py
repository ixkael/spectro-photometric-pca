# -*- coding: utf-8 -*-

import numpy as onp

import jax
from jax import partial, jit
import jax.numpy as np

from gasp.pca_utils_jx import *
from gasp.marginallikelihoods_jx import *

from jax.nn import sigmoid

def chebychevPolynomials(n_poly, n_pix_spec):
    x = np.linspace(-1.0, 1.0, n_pix_spec)
    res = np.vstack(
        [x * 0 + 1, x, 2 * x ** 2 - 1, 4 * x ** 3 - 3 * x, 8 * x ** 4 - 8 * x ** 2 + 1]
    )  # (n_poly, n_pix_spec)
    return res[0:n_poly, :]


class PriorModel:
    """"""

    def __init__(self, n_components):
        self.n_components = n_components

    def random(self, key):
        params_means = 0 * jax.random.normal(key, (self.n_components, 1)) + 0
        params_loginvvar = 0 * jax.random.normal(key, (self.n_components, 1)) + 0*np.log(10) # how many log10 octaves
        params = np.hstack([params_means, params_loginvvar])
        return params

    @staticmethod
    def get_mean_at_z(params, redshifts):
        cst = np.ones((redshifts.size, 1)) * params[None, :, 0]
        #slp = redshifts[:, None] * (params[None, :, 1] - params[None, :, 0])
        return cst#+ slp

    @staticmethod
    def get_loginvvar_at_z(params, redshifts):
        cst = np.ones((redshifts.size, 1)) * params[None, :, 1]
        #slp = redshifts[:, None] * (params[None, :, 3] - params[None, :, 2])
        return cst# + slp


@jit
def bayesianpca_speconly_explicit(
    components_spec,  # [n_obj, n_components, nspec]
    polynomials_spec,  # [n_poly, nspec]
    spec,  # [n_obj, nspec]
    spec_invvar,  # [n_obj, nspec]
    spec_loginvvar,  # [n_obj, nspec]
    components_prior_mean,  # [n_obj, n_components]
    components_prior_loginvvar,  # [n_obj, n_components]
    polynomials_prior_mean,  # [n_obj, n_components] or [n_poly]
    polynomials_prior_loginvvar,  # [n_obj, n_components] or [n_poly]
):

    n_obj, nspec = np.shape(spec)[0], np.shape(spec)[1]
    n_components = np.shape(components_spec)[1]
    n_poly = np.shape(polynomials_spec)[0]

    components_spec_all = np.concatenate(
        [components_spec, polynomials_spec[None, :, :] * np.ones((n_obj, 1, nspec))],
        axis=-2,
    )  # [n_obj, n_components+n_poly, nspec]

    # if shape is [n_poly] instead of [n_obj, n_components]
    mu = np.concatenate(
        [components_prior_mean, polynomials_prior_mean[None, :] * np.ones((n_obj, 1))],
        axis=-1,
    )
    muinvvar = 1 / sigmoid(np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, :] * np.ones((n_obj, 1)),
        ],
        axis=-1,
    ))
    #muinvvar = np.exp(logmuinvvar)
    # if shape is [n_obj, n_components] instead of [n_poly]
    #mu = polynomials_prior_mean
    #muinvvar = np.exp(polynomials_prior_loginvvar)
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
def bayesianpca_specandphot_explicit(
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
    polynomials_prior_mean,  # [n_obj, n_components] or [n_poly]
    polynomials_prior_loginvvar,  # [n_obj, n_components] or [n_poly]
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

    # if shape is [n_poly] instead of [n_obj, n_components]
    mu = np.concatenate(
        [components_prior_mean, polynomials_prior_mean[None, :] * np.ones((n_obj, 1))],
        axis=-1,
    )
    muinvvar = 1 / sigmoid(np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, :] * np.ones((n_obj, 1)),
            ],
        axis=-1,
    ))
    #muinvvar = np.exp(logmuinvvar)
    # if shape is [n_obj, n_components] instead of [n_poly]
    #mu = polynomials_prior_mean
    #muinvvar = np.exp(polynomials_prior_loginvvar)
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


def loss_speconly(
    params,
    data_batch,
    data_aux,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):
    (logfml, _, _, _, _, _) = bayesianpca_speconly(
        params,
        data_batch,
        data_aux,
        n_components,
        n_pix_spec,
        opt_basis,
        opt_priors,
    )
    return -np.sum(logfml)


def loss_specandphot(
    params,
    data_batch,
    data_aux,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):
    (logfml, _, _, _, _, _) = bayesianpca_specandphot(
        params,
        data_batch,
        data_aux,
        n_components,
        n_pix_spec,
        opt_basis,
        opt_priors,
    )
    return -np.sum(logfml)


def bayesianpca_speconly(
    params,
    data_batch,
    data_aux,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):

    if opt_basis and opt_priors:
        polynomials_spec = data_aux
        pcacomponents_speconly = params[0]
        priors_speconly = [params[1], params[2], params[3]]
    if opt_basis and not opt_priors:
        priors_speconly, polynomials_spec = data_aux
        pcacomponents_speconly = params[0]
    if not opt_basis and opt_priors:
        pcacomponents_speconly, polynomials_spec = data_aux
        priors_speconly = params

    (
        components_prior_params_speconly,
        polynomials_prior_mean_speconly,
        polynomials_prior_loginvvar_speconly,
    ) = priors_speconly

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
        batch_interpweights
    ) = data_batch

    # ones = np.ones((bs, 1))
    # batch_index_wave = np.zeros((bs,), int)

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
    ) = bayesianpca_speconly_explicit(
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
        components_phot_speconly[:, :, :] * thetamap_speconly[:, 0:n_components, None],
        axis=1,
    )
    _, ellfactors, _ = logmarglike_lineargaussianmodel_onetransfer_jitvmap(
        photmod_map_speconly[:, None, :], phot, phot_invvar, phot_loginvvar
    )
    return (
        logfml_speconly,
        thetamap_speconly,
        thetastd_speconly,
        specmod_map_speconly,
        photmod_map_speconly,
        np.ravel(ellfactors),
    )


def bayesianpca_specandphot(
    params,
    data_batch,
    data_aux,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):

    if opt_basis and opt_priors:
        polynomials_spec = data_aux
        pcacomponents_specandphot = params[0]
        priors_specandphot = [params[1], params[2], params[3]]
    if opt_basis and not opt_priors:
        priors_specandphot, polynomials_spec = data_aux
        pcacomponents_specandphot = params[0]
    if not opt_basis and opt_priors:
        pcacomponents_specandphot, polynomials_spec = data_aux
        priors_specandphot = params

    (
        components_prior_params_specandphot,
        polynomials_prior_mean_specandphot,
        polynomials_prior_loginvvar_specandphot,
    ) = priors_specandphot

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
        batch_interpweights
    ) = data_batch

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

    # only if polynomial prior is redshift dependent
    #polynomials_prior_mean_specandphot = PriorModel.get_loginvvar_at_z(
    #    components_prior_params_specandphot, batch_redshifts
    #)
    #polynomials_prior_loginvvar_specandphot = PriorModel.get_mean_at_z(
    #    components_prior_params_specandphot, batch_redshifts
    #)

    (_, thetamap_speconly, _, _,) = bayesianpca_speconly_explicit(
        pcacomponents_specandphot_atz,  # [n_obj, n_components, nspec]
        polynomials_spec,  # [n_poly, nspec]
        spec,  # [n_obj, nspec]
        spec_invvar,  # [n_obj, nspec]
        spec_loginvvar,  # [n_obj, nspec]
        components_prior_mean_specandphot,  # [n_obj, n_components]
        components_prior_loginvvar_specandphot,  # [n_obj, n_components]
        polynomials_prior_mean_specandphot,
        polynomials_prior_loginvvar_specandphot,
    )
    # ellfactors = np.ones_like(batch_redshifts)
    photmod_map_speconly = np.sum(
        components_phot_specandphot[:, :, :]
        * thetamap_speconly[:, 0:n_components, None],
        axis=1,
    )
    _, ellfactors, _ = logmarglike_lineargaussianmodel_onetransfer_jitvmap(
        photmod_map_speconly[:, None, :], phot, phot_invvar, phot_loginvvar
    )

    (
        logfml_specandphot,
        thetamap_specandphot,
        thetastd_specandphot,
        specmod_map_specandphot,
        photmod_map_specandphot,
    ) = bayesianpca_specandphot_explicit(
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
        np.ravel(ellfactors),
    )


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

        self.pcacomponents = np.load(
            self.prefix + "pcacomponents" + self.suffix + ".npy"
        )
        self.components_prior_params = np.load(
            self.prefix + "components_prior_params" + self.suffix + ".npy",
        )
        self.polynomials_prior_mean = np.load(
            self.prefix + "polynomials_prior_mean" + self.suffix + ".npy",
        )
        self.polynomials_prior_loginvvar = np.load(
            self.prefix + "polynomials_prior_loginvvar" + self.suffix + ".npy",
        )

    def init_params(
        self, key, n_components, n_poly, n_pix_sed, opt_basis=True, opt_priors=True
    ):

        self.pcacomponents_prior = PriorModel(n_components)

        self.opt_basis, self.opt_priors = opt_basis, opt_priors

        self.pcacomponents = jax.random.normal(key, (n_components, n_pix_sed))
        self.components_prior_params = self.pcacomponents_prior.random(key)
        self.polynomials_prior_mean = 0 * jax.random.normal(key, (n_poly,)) + 0
        self.polynomials_prior_loginvvar = 0 * jax.random.normal(key, (n_poly,)) + 2*np.log(10) # how many log10 octaves

        return self.pcacomponents_prior

    def get_params_opt(self):
        arr = []
        if self.opt_basis:
            arr = [self.pcacomponents]
        if self.opt_priors:
            arr += [
                self.components_prior_params,
                self.polynomials_prior_mean,
                self.polynomials_prior_loginvvar,
            ]
        return arr

    def get_params_nonopt(self):
        arr = []
        if not self.opt_basis and self.opt_priors:
            arr = [self.pcacomponents]
        if self.opt_basis and not self.opt_priors:
            arr = [
                self.components_prior_params,
                self.polynomials_prior_mean,
                self.polynomials_prior_loginvvar,
            ]
        if not self.opt_basis and not self.opt_priors:
            arr = [self.pcacomponents] + [
                self.components_prior_params,
                self.polynomials_prior_mean,
                self.polynomials_prior_loginvvar,
            ]
        return arr

    def set_params(self, params):
        off = 0
        if self.opt_basis:
            self.pcacomponents = params[off]
            off += 1
        if self.opt_priors:
            self.components_prior_params = params[off]
            self.polynomials_prior_mean = params[off + 1]
            self.polynomials_prior_loginvvar = params[off + 2]
