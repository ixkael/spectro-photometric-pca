# -*- coding: utf-8 -*-

import numpy as onp

import jax
from jax import partial, jit
import jax.numpy as np

from gasp.pca_utils_jx import *
from gasp.marginallikelihoods_jx import *

from jax.nn import sigmoid
from jax.scipy.special import logsumexp


def batch_indices(start_indices, n_components, npix):
    nobj = start_indices.shape[0]

    indices_2d = start_indices[:, None] + np.arange(npix)[None, :]

    indices_0 = np.arange(n_components)[None, :, None] * np.ones(
        (nobj, n_components, npix), dtype=int
    )
    indices_1 = indices_2d[:, None, :] * np.ones((1, n_components, npix), dtype=int)

    return indices_0, indices_1


def batch_indices_2(start_indices, n_components, npix):
    nobj, n_pix_spec = start_indices.shape

    indices_0 = np.arange(n_components)[None, :, None] * np.ones(
        (nobj, n_components, npix), dtype=int
    )
    indices_1 = start_indices[:, None, :] * np.ones((1, n_components, npix), dtype=int)

    return indices_0, indices_1


def batch_indices_3(start_indices, n_archetypes, n_components, npix):
    nobj, n_pix_spec = start_indices.shape

    indices_0 = np.arange(n_archetypes)[None, :, None, None] * np.ones(
        (nobj, n_archetypes, n_components, npix), dtype=int
    )
    indices_1 = np.arange(n_components)[None, None, :, None] * np.ones(
        (nobj, n_archetypes, n_components, npix), dtype=int
    )
    indices_2 = start_indices[:, None, None, :] * np.ones(
        (1, n_archetypes, n_components, npix), dtype=int
    )

    return indices_0, indices_1, indices_2


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
        self,
        key,
        n_archetypes,
        n_components,
        n_poly,
        n_pix_sed,
        opt_basis=True,
        opt_priors=True,
    ):

        self.pcacomponents_prior = PriorModel(n_archetypes, n_components)

        self.opt_basis, self.opt_priors = opt_basis, opt_priors

        self.pcacomponents = jax.random.normal(
            key, (n_archetypes, n_components, n_pix_sed)
        )
        self.components_prior_params = self.pcacomponents_prior.random(key)
        self.polynomials_prior_mean = 0 * jax.random.normal(key, (n_poly,)) + 0
        self.polynomials_prior_loginvvar = 0 * jax.random.normal(
            key, (n_poly,)
        ) + 0 * np.log(
            10
        )  # how many log10 octaves

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


def chebychevPolynomials(n_poly, n_pix_spec):
    x = np.linspace(-1.0, 1.0, n_pix_spec)
    res = np.vstack(
        [x * 0 + 1, x, 2 * x ** 2 - 1, 4 * x ** 3 - 3 * x, 8 * x ** 4 - 8 * x ** 2 + 1]
    )  # (n_poly, n_pix_spec)
    return res[0:n_poly, :]


class PriorModel:
    """"""

    def __init__(self, n_archetypes, n_components):
        self.n_archetypes = n_archetypes
        self.n_components = n_components

    def random(self, key):
        params_means = 0 * jax.random.normal(
            key, (self.n_archetypes, self.n_components, 2)
        )
        params_loginvvar = 0 * jax.random.normal(
            key, (self.n_archetypes, self.n_components, 2)
        )
        params = np.concatenate([params_means, params_loginvvar], axis=-1)
        return params

    @staticmethod
    def get_mean_at_z(params, redshifts):
        cst = np.ones((redshifts.size, 1, 1)) * params[None, :, :, 0]
        slp = redshifts[:, None, None] * (params[None, :, :, 1] - params[None, :, :, 0])
        return cst + slp
        return (
            params[None, :, :, 0]
            * (redshifts[:, None, None] - params[None, :, :, 1])
            * (redshifts[:, None, None] - params[None, :, :, 2])
        )

    @staticmethod
    def get_loginvvar_at_z(params, redshifts):
        expparams = np.exp(params)
        cst = np.ones((redshifts.size, 1, 1)) * expparams[None, :, :, 2]
        slp = redshifts[:, None, None] * (
            expparams[None, :, :, 3] - expparams[None, :, :, 2]
        )
        stddev = cst + slp
        # stddev = (
        #    params[None, :, 3]
        #    * (redshifts[:, None] - params[None, :, 4])
        #    * (redshifts[:, None] - params[None, :, 5])
        # )
        return np.log(
            stddev ** -2
        )  # because want the stddev to be linear, not the variance


def map_to_minusone_one(x):
    return 2 * sigmoid(x) - 1


def loss_pca_photonly(
    params,
    data_batch,
    data_aux,
    n_components,
    opt_basis,
    opt_priors,
    regularization,
):
    (logfml, _, _, _) = bayesianpca_photonly(
        params,
        data_batch,
        data_aux,
        n_components,
        opt_basis,
        opt_priors,
    )
    if opt_basis and opt_priors:
        pcacomponents = params[0]
        pcacomponents_init = data_aux[0]
    if opt_basis and not opt_priors:
        pcacomponents = params[0]
        pcacomponents_init = data_aux[1]
    if not opt_basis and opt_priors:
        pcacomponents = data_aux[0]
        pcacomponents_init = data_aux[1]
    diff = pcacomponents - pcacomponents_init
    return -np.sum(logfml)  # + np.sum(diff ** 2) * regularization


@jit
def bayesianpca_speconly_explicit(
    components_spec,  # [n_obj, n_archetypes, n_components, nspec]
    polynomials_spec,  # [n_poly, nspec]
    spec,  # [n_obj, nspec]
    spec_invvar,  # [n_obj, nspec]
    spec_loginvvar,  # [n_obj, nspec]
    components_prior_mean,  # [n_obj, n_archetypes, n_components]
    components_prior_loginvvar,  # [n_obj, n_archetypes, n_components]
    polynomials_prior_mean,  # [n_poly]
    polynomials_prior_loginvvar,  # [n_poly]
):

    n_obj, n_archetypes, n_components, nspec = np.shape(components_spec)
    n_poly = np.shape(polynomials_spec)[0]

    components_spec_all = np.concatenate(
        [
            components_spec,
            polynomials_spec[None, None, :, :]
            * np.ones((n_obj, n_archetypes, 1, nspec)),
        ],
        axis=-2,
    )  # [n_obj, n_archetypes, n_components+n_poly, nspec]

    # if shape is [n_poly] instead of [n_obj, n_components]
    mu = np.concatenate(
        [
            components_prior_mean,
            polynomials_prior_mean[None, None, :] * np.ones((n_obj, n_archetypes, 1)),
        ],
        axis=-1,
    )
    logmuinvvar = np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, None, :]
            * np.ones((n_obj, n_archetypes, 1)),
        ],
        axis=-1,
    )
    muinvvar = np.exp(logmuinvvar)
    # if shape is [n_obj, n_components] instead of [n_poly]
    # mu = polynomials_prior_mean
    # muinvvar = np.exp(polynomials_prior_loginvvar)
    # logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        thetamap,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_twotransfers_jitvmapvmap(
        components_spec_all,
        spec[:, None, :] * np.ones((1, n_archetypes, 1)),
        spec_invvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        spec_loginvvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        mu,
        muinvvar,
        logmuinvvar,
    )

    thetastd = np.diagonal(theta_cov, axis1=2, axis2=3) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all * thetamap[:, :, :, None], axis=-2)
    # chi2_spec = np.sum((specmod_map - spec) ** 2 * spec_invvar, axis=-1)

    return (logfml, thetamap, thetastd, specmod_map)


@jit
def bayesianpca_specandphot_explicit(
    components_spec,  # [n_obj, n_archetypes, n_components, nspec]
    components_phot,  # [n_obj, n_archetypes, n_components, nphot]
    polynomials_spec,  # [n_poly, nspec]
    ellfactors,  # [n_obj, n_archetypes]
    spec,  # [n_obj, nspec]
    spec_invvar,  # [n_obj, nspec]
    spec_loginvvar,  # [n_obj, nspec]
    phot,  # [n_obj, nphot]
    phot_invvar,  # [n_obj, nphot]
    phot_loginvvar,  # [n_obj, nphot]
    components_prior_mean,  # [n_obj, n_archetypes, n_components]
    components_prior_loginvvar,  # [n_obj, n_archetypes, n_components]
    polynomials_prior_mean,  # [n_poly]
    polynomials_prior_loginvvar,  # [n_poly]
):

    n_obj, n_archetypes, n_components, nspec = np.shape(components_spec)
    n_poly = np.shape(polynomials_spec)[0]
    nphot = np.shape(phot)[1]

    components_spec_all = np.concatenate(
        [
            components_spec,
            polynomials_spec[None, :, :] * np.ones((n_obj, n_archetypes, 1, nspec)),
        ],
        axis=-2,
    )  # [n_obj, n_archetypes, n_components+n_poly, nspec]

    components_phot_all = np.concatenate(
        [components_phot, np.zeros((n_obj, n_archetypes, n_poly, nphot))], axis=2
    )  # [n_obj,n_archetypes,  n_components+n_poly, nphot]

    # if shape is [n_poly] instead of [n_obj, n_components]
    mu = np.concatenate(
        [
            components_prior_mean,
            polynomials_prior_mean[None, None, :] * np.ones((n_obj, n_archetypes, 1)),
        ],
        axis=-1,
    )
    logmuinvvar = np.concatenate(
        [
            components_prior_loginvvar,
            polynomials_prior_loginvvar[None, None, :]
            * np.ones((n_obj, n_archetypes, 1)),
        ],
        axis=-1,
    )
    muinvvar = np.exp(logmuinvvar)
    # if shape is [n_obj, n_components] instead of [n_poly]
    # mu = polynomials_prior_mean
    # muinvvar = np.exp(polynomials_prior_loginvvar)
    # logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        thetamap,
        theta_cov,
    ) = logmarglike_lineargaussianmodel_threetransfers_jitvmapvmap(
        ellfactors,
        components_spec_all,
        components_phot_all,
        spec[:, None, :] * np.ones((1, n_archetypes, 1)),
        spec_invvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        spec_loginvvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_invvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_loginvvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        mu,
        muinvvar,
        logmuinvvar,
    )

    thetastd = np.diagonal(theta_cov, axis1=2, axis2=3) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all * thetamap[:, :, :, None], axis=-2)
    photmod_map = np.sum(components_phot_all * thetamap[:, :, :, None], axis=-2)

    return (logfml, thetamap, thetastd, specmod_map, photmod_map)


def bayesianpca_speconly(
    params,
    data_batch,
    data_aux,
    n_archetypes,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):

    if opt_basis and opt_priors:
        polynomials_spec = data_aux[0]
        pcacomponents_speconly = params[0]
        priors_speconly = [params[1], params[2], params[3]]
    if opt_basis and not opt_priors:
        priors_speconly, polynomials_spec = data_aux[0], data_aux[1]
        pcacomponents_speconly = params[0]
    if not opt_basis and opt_priors:
        pcacomponents_speconly, polynomials_spec = data_aux[0], data_aux[1]
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
        batch_interpweights,
    ) = data_batch

    indices_0, indices_1, indices_2 = batch_indices_3(
        batch_interprightindices, n_archetypes, n_components, n_pix_spec
    )

    # pcacomponents_speconly is (n_archetypes, n_components, n_pix_sed)
    # batch_index_wave is (nobj, )
    # batch_interprightindices is (nobj, n_pix_spec)
    # pcacomponents_speconly_atz is (nobj, n_archetypes, n_components, n_pix_spec)
    pcacomponents_speconly_atz = (
        batch_interpweights[:, None, None, :]
        * pcacomponents_speconly[indices_0, indices_1, indices_2 - 1]
        + (1 - batch_interpweights[:, None, None, :])
        * pcacomponents_speconly[indices_0, indices_1, indices_2]
    )
    components_phot_speconly = np.sum(
        pcacomponents_speconly[None, :, :, :, None]
        * batch_transferfunctions[:, None, None, :, :],
        axis=3,
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
        pcacomponents_speconly_atz,  # [n_obj, n_archetypes, n_components, nspec]
        polynomials_spec,  # [n_poly, nspec]
        spec,  # [n_obj, nspec]
        spec_invvar,  # [n_obj, nspec]
        spec_loginvvar,  # [n_obj, nspec]
        components_prior_mean_speconly,  # [n_obj, n_archetypes, n_components]
        components_prior_loginvvar_speconly,  # [n_obj, n_archetypes, n_components]
        polynomials_prior_mean_speconly,
        polynomials_prior_loginvvar_speconly,
    )
    photmod_map_speconly = np.sum(
        components_phot_speconly[:, :, :, :]
        * thetamap_speconly[:, :, 0:n_components, None],
        axis=-2,
    )  # [n_obj, n_phot]
    _, ellfactors, _ = logmarglike_lineargaussianmodel_onetransfer_jitvmapvmap(
        photmod_map_speconly[:, :, None, :],
        phot[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_invvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_loginvvar[:, None, :] * np.ones((1, n_archetypes, 1)),
    )
    return (
        logfml_speconly,
        thetamap_speconly,
        thetastd_speconly,
        specmod_map_speconly,
        photmod_map_speconly,
        ellfactors,
    )


def bayesianpca_specandphot(
    params,
    data_batch,
    data_aux,
    n_archetypes,
    n_components,
    n_pix_spec,
    opt_basis,
    opt_priors,
):

    if opt_basis and opt_priors:
        polynomials_spec = data_aux[0]
        pcacomponents_specandphot = params[0]
        priors_specandphot = [params[1], params[2], params[3]]
    if opt_basis and not opt_priors:
        priors_specandphot, polynomials_spec = data_aux[0], data_aux[1]
        pcacomponents_specandphot = params[0]
    if not opt_basis and opt_priors:
        pcacomponents_specandphot, polynomials_spec = data_aux[0], data_aux[1]
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
        batch_interpweights,
    ) = data_batch

    indices_0, indices_1, indices_2 = batch_indices_3(
        batch_interprightindices, n_archetypes, n_components, n_pix_spec
    )

    # pcacomponents_speconly is (n_archetypes, n_components, n_pix_sed)
    # batch_index_wave is (nobj, )
    # batch_interprightindices is (nobj, n_pix_spec)
    # pcacomponents_speconly_atz is (nobj, n_archetypes, n_components, n_pix_spec)
    pcacomponents_specandphot_atz = (
        batch_interpweights[:, None, None, :]
        * pcacomponents_specandphot[indices_0, indices_1, indices_2 - 1]
        + (1 - batch_interpweights[:, None, None, :])
        * pcacomponents_specandphot[indices_0, indices_1, indices_2]
    )
    components_phot_specandphot = np.sum(
        pcacomponents_specandphot[None, :, :, :, None]
        * batch_transferfunctions[:, None, None, :, :],
        axis=3,
    )
    components_prior_mean_specandphot = PriorModel.get_mean_at_z(
        components_prior_params_specandphot, batch_redshifts
    )
    components_prior_loginvvar_specandphot = PriorModel.get_loginvvar_at_z(
        components_prior_params_specandphot, batch_redshifts
    )

    # only if polynomial prior is redshift dependent
    # polynomials_prior_mean_specandphot = PriorModel.get_loginvvar_at_z(
    #    components_prior_params_specandphot, batch_redshifts
    # )
    # polynomials_prior_loginvvar_specandphot = PriorModel.get_mean_at_z(
    #    components_prior_params_specandphot, batch_redshifts
    # )

    (_, thetamap_speconly, _, _,) = bayesianpca_speconly_explicit(
        pcacomponents_specandphot_atz,  # [n_obj, n_archetypes, n_components, nspec]
        polynomials_spec,  # [n_poly, nspec]
        spec,  # [n_obj, nspec]
        spec_invvar,  # [n_obj, nspec]
        spec_loginvvar,  # [n_obj, nspec]
        components_prior_mean_specandphot,  # [n_obj, n_archetypes, n_components]
        components_prior_loginvvar_specandphot,  # [n_obj, n_archetypes, n_components]
        polynomials_prior_mean_specandphot,
        polynomials_prior_loginvvar_specandphot,
    )
    photmod_map_speconly = np.sum(
        components_phot_specandphot[:, :, :, :]
        * thetamap_speconly[:, :, 0:n_components, None],
        axis=2,
    )
    _, ellfactors, _ = logmarglike_lineargaussianmodel_onetransfer_jitvmapvmap(
        photmod_map_speconly[:, :, None, :],
        phot[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_invvar[:, None, :] * np.ones((1, n_archetypes, 1)),
        phot_loginvvar[:, None, :] * np.ones((1, n_archetypes, 1)),
    )

    (
        logfml_specandphot,
        thetamap_specandphot,
        thetastd_specandphot,
        specmod_map_specandphot,
        photmod_map_specandphot,
    ) = bayesianpca_specandphot_explicit(
        pcacomponents_specandphot_atz,  # [n_obj, n_archetypes, n_components, nspec]
        components_phot_specandphot,  # [n_obj, n_archetypes, n_components, nphot]
        polynomials_spec,  # [n_poly, nspec]
        ellfactors,  # [n_obj, n_archetypes]
        spec,  # [n_obj, nspec]
        spec_invvar,  # [n_obj, nspec]
        spec_loginvvar,  # [n_obj, nspec]
        phot,  # [n_obj, nphot]
        phot_invvar,  # [n_obj, nphot]
        phot_loginvvar,  # [n_obj, nphot]
        components_prior_mean_specandphot,  # [n_obj, n_archetypes, n_components]
        components_prior_loginvvar_specandphot,  # [n_obj, n_archetypes, n_components]
        polynomials_prior_mean_specandphot,
        polynomials_prior_loginvvar_specandphot,
    )
    return (
        logfml_specandphot,
        thetamap_specandphot,
        thetastd_specandphot,
        specmod_map_specandphot,
        photmod_map_specandphot,
        ellfactors,
    )


def bayesianpca_photonly(
    params,
    data_batch,
    data_aux,
    n_components,
    opt_basis,
    opt_priors,
):

    if opt_basis and opt_priors:
        pcacomponents_photonly = params[0]
        components_prior_params_photonly = params[1]
    if opt_basis and not opt_priors:
        components_prior_params_photonly = data_aux[0]
        pcacomponents_photonly = params[0]
    if not opt_basis and opt_priors:
        pcacomponents_photonly = data_aux[0]
        components_prior_params_photonly = params[0]

    (
        si,
        bs,
        phot,
        phot_invvar,
        phot_loginvvar,
        batch_redshifts,
        transferfunctions,
        batch_interprightindices_transfer,
        batch_interpweights_transfer,
    ) = data_batch

    components_phot_photonly = np.sum(
        pcacomponents_photonly[None, :, :, None] * transferfunctions[:, None, :, :],
        axis=2,
    )  # [n_z_transfer, n_components, n_phot]

    components_phot_photonly_obj = np.take(
        components_phot_photonly, batch_interprightindices_transfer, axis=0
    )
    components_phot_photonly_atz = (
        batch_interpweights_transfer[:, None, None] * components_phot_photonly_obj
        + (1 - batch_interpweights_transfer[:, None, None])
        * components_phot_photonly_obj
    )

    components_prior_mean_photonly = PriorModel.get_mean_at_z(
        components_prior_params_photonly, batch_redshifts
    )
    components_prior_loginvvar_photonly = PriorModel.get_loginvvar_at_z(
        components_prior_params_photonly, batch_redshifts
    )
    components_prior_invvar_photonly = np.exp(components_prior_loginvvar_photonly)
    (
        logfml_photonly,
        thetamap_photonly,
        theta_cov_photonly,
    ) = logmarglike_lineargaussianmodel_twotransfers_jitvmap(
        components_phot_photonly_atz,  # [n_obj, n_components, nphot]
        phot,  # [n_obj, nphot]
        phot_invvar,  # [n_obj, nphot]
        phot_loginvvar,  # [n_obj, nphot]
        components_prior_mean_photonly,
        components_prior_invvar_photonly,
        components_prior_loginvvar_photonly,
    )
    # ellfactors = np.ones_like(batch_redshifts)
    photmod_map_photonly = np.sum(
        components_phot_photonly_atz * thetamap_photonly[:, :, None],
        axis=1,
    )
    thetastd_photonly = np.diagonal(theta_cov_photonly, axis1=1, axis2=2) ** 0.5

    return (
        logfml_photonly,
        thetamap_photonly,
        thetastd_photonly,
        photmod_map_photonly,
    )
