# -*- coding: utf-8 -*-

import numpy as onp

import jax
from jax import jit
import jax.numpy as np

from gasp.pca_utils_jx import *
from gasp.marginallikelihoods_jx import *


class PriorModel:
    """"""

    def __init__(self, num_components):
        self.num_components = num_components
        self.params_shape = (self.num_components, 2)

    def random(self, key):
        params = jax.random.normal(key, self.params_shape)
        return params

    def get_mean_at_z(self, params, redshifts):
        return np.ones((redshifts.size, 1)) * params[None, :, 0]

    def get_loginvvar_at_z(self, params, redshifts):
        return np.ones((redshifts.size, 1)) * params[None, :, 1]


@jit
def bayesianpca_speconly(
    components_spec,  # [nobj, nlatent, nspec]
    polynomials_spec,  # [npoly, nspec]
    spec,  # [nobj, nspec]
    spec_invvar,  # [nobj, nspec]
    spec_loginvvar,  # [nobj, nspec]
    components_prior_mean,  # [nobj, nlatent]
    components_prior_loginvvar,  # [nobj, nlatent]
    polynomials_prior_mean,  # [nobj, npoly]
    polynomials_prior_loginvvar,  # [nobj, npoly]
):

    nobj, nspec = np.shape(spec)[0], np.shape(spec)[1]
    nlatent = np.shape(components_spec)[1]
    npoly = np.shape(polynomials_spec)[0]

    components_spec_all = np.concat(
        [components_spec, polynomials_spec[None, :, :] * np.ones((nobj, 1, nspec))],
        axis=-2,
    )  # [nobj, nlatent+npoly, nspec]

    mu = np.concat([components_prior_mean, polynomials_prior_mean], axis=-1)
    logmuinvvar = np.concat(
        [polynomials_prior_loginvvar, components_prior_loginvvar], axis=-1
    )
    muinvvar = np.exp(logmuinvvar)

    logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        theta_map,
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

    theta_std = np.diagonal(theta_cov, axis1=1, axis2=2) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all[:, :, :] * theta_map[:, :, None], axis=1)
    # chi2_spec = np.sum((specmod_map - spec) ** 2 * spec_invvar, axis=-1)

    return (logfml, theta_map, theta_std, specmod_map)


@jit
def bayesianpca_specandphot(
    components_spec,  # [nobj, nlatent, nspec]
    components_phot,  # [nobj, nlatent, nphot]
    polynomials_spec,  # [npoly, nspec]
    ellfactors,  # [nobj, ]
    spec,  # [nobj, nspec]
    spec_invvar,  # [nobj, nspec]
    spec_loginvvar,  # [nobj, nspec]
    phot,  # [nobj, nphot]
    phot_invvar,  # [nobj, nphot]
    phot_loginvvar,  # [nobj, nphot]
    components_prior_mean,  # [nobj, nlatent]
    components_prior_loginvvar,  # [nobj, nlatent]
    polynomials_prior_mean,  # [nobj, npoly]
    polynomials_prior_loginvvar,  # [nobj, npoly]
):

    nobj, nspec = np.shape(spec)[0], np.shape(spec)[1]
    nlatent = np.shape(components_spec)[1]
    npoly = np.shape(polynomials_spec)[0]
    nphot = np.shape(phot)[1]

    components_spec_all = np.concat(
        [components_spec, polynomials_spec[None, :, :] * np.ones((nobj, 1, nspec))],
        axis=-2,
    )  # [nobj, nlatent+npoly, nspec]

    zeros = np.zeros((nobj, npoly, nphot), dtype=T)
    components_phot_all = np.concat(
        [components_phot, zeros], axis=1
    )  # [nobj, nlatent+npoly, nphot]

    mu = np.concat([components_prior_mean, polynomials_prior_mean], axis=-1)
    logmuinvvar = np.concat(
        [polynomials_prior_loginvvar, components_prior_loginvvar], axis=-1
    )
    muinvvar = np.exp(logmuinvvar)

    logmuinvvar = np.log(muinvvar)  # Assume no mask in last dimension
    (
        logfml,
        theta_map,
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

    theta_std = np.diagonal(theta_cov, axis1=1, axis2=2) ** 0.5

    # Produce best fit models
    specmod_map = np.sum(components_spec_all[:, :, :] * theta_map[:, :, None], axis=1)
    photmod_map = np.sum(components_phot_all[:, :, :] * theta_map[:, :, None], axis=1)
    # chi2_spec = np.sum((specmod_map - spec) ** 2 * spec_invvar, axis=-1)
    # chi2_phot = np.sum((photmod_map - phot) ** 2 * phot_invvar, axis=-1)

    return (logfml, theta_map, theta_std, specmod_map, photmod_map)


@jit
def bayesianpca_spec_and_specandphot(params_list, data_batch, aux_data):

    (
        all_ells,
        pcacomponents_speconly,
        components_prior_params_speconly,
        polynomials_prior_mean_speconly,
        polynomials_prior_loginvvar_speconly,
        pcacomponents_specandphot,
        components_prior_params_specandphot,
        polynomials_prior_mean_specandphot,
        polynomials_prior_loginvvar_specandphot,
    ) = params_list

    (
        si,
        batch_index_wave,
        batch_index_transfer_redshift,
        spec,
        spec_invvar,
        spec_loginvvar,
        batch_spec_mask,
        specphotscaling,
        phot,
        phot_invvar,
        phot_loginvvar,
        batch_redshifts,
        bs,
        batch_transferfunctions,
        batch_index_wave_ext,
    ) = data_batch

    ellfactors = all_ells[si : si + bs]

    (polynomials_spec, specwavesize, components_prior) = aux_data

    pcacomponents_speconly_atz = take_batch(
        pcacomponents_speconly, batch_index_wave, npix
    )
    components_prior_mean_speconly = components_prior.get_mean_at_z(
        components_prior_params_speconly, batch_redshifts
    )
    components_prior_loginvvar_speconly = components_prior.get_loginvvar_at_z(
        components_prior_params_speconly, batch_redshifts
    )
    (
        logfml_speconly,
        theta_map_speconly,
        theta_std_speconly,
        specmod_map_speconly,
    ) = bayesianpca_speconly(
        pcacomponents_speconly_atz,  # [nobj, nlatent, nspec]
        polynomials_spec,  # [npoly, nspec]
        spec,  # [nobj, nspec]
        spec_invvar,  # [nobj, nspec]
        spec_loginvvar,  # [nobj, nspec]
        components_prior_mean_speconly,  # [nobj, nlatent]
        components_prior_loginvvar_speconly,  # [nobj, nlatent]
        polynomials_prior_mean_speconly,  # [nobj, npoly]
        polynomials_prior_loginvvar_speconly,  # [nobj, npoly]
    )

    pcacomponents_specandphot_atz = take_batch(
        pcacomponents_specandphot, batch_index_wave, npix
    )
    components_phot = np.sum(
        pcacomponents_specandphot[None, :, :, None]
        * batch_transferfunctions[:, None, :, :],
        axis=2,
    )
    components_prior_mean_specandphot = components_prior.get_mean_at_z(
        components_prior_params_specandphot, batch_redshifts
    )
    components_prior_loginvvar_specandphot = components_prior.get_loginvvar_at_z(
        components_prior_params_specandphot, batch_redshifts
    )
    (
        logfml_specandphot,
        theta_map_specandphot,
        theta_std_specandphot,
        specmod_map_specandphot,
    ) = bayesianpca_specandphot(
        pcacomponents_specandphot_atz,  # [nobj, nlatent, nspec]
        components_phot,  # [nobj, nlatent, nphot]
        polynomials_spec,  # [npoly, nspec]
        ellfactors,  # [nobj, ]
        spec,  # [nobj, nspec]
        spec_invvar,  # [nobj, nspec]
        spec_loginvvar,  # [nobj, nspec]
        phot,  # [nobj, nphot]
        phot_invvar,  # [nobj, nphot]
        phot_loginvvar,  # [nobj, nphot]
        components_prior_mean_specandphot,  # [nobj, nlatent]
        components_prior_loginvvar_specandphot,  # [nobj, nlatent]
        polynomials_prior_mean_specandphot,  # [nobj, npoly]
        polynomials_prior_loginvvar_specandphot,  # [nobj, npoly]
    )
    return (
        logfml_speconly,
        theta_map_speconly,
        theta_std_speconly,
        specmod_map_speconly,
        logfml_specandphot,
        theta_map_specandphot,
        theta_std_specandphot,
        specmod_map_specandphot,
    )


@jit
def loss_spec_and_specandphot(params_list, data_batch, aux_data):
    (
        logfml_speconly,
        _,
        _,
        _,
        logfml_specandphot,
        _,
        _,
        _,
    ) = bayesianpca_spec_and_specandphot(params_list, data_batch, aux_data)
    return -np.sum(logfml_specandphot + logfml_speconly)


@jit
def update(step, opt_state, data):
    params = get_params(opt_state)
    value, grads = jax.value_and_grad(loss_spec_and_specandphot)(params, data)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state
