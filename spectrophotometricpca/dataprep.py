# -*- coding: utf-8 -*-

import numpy as np

# from dustmaps.sfd import SFDQuery
# from astropy.coordinates import SkyCoord
from astropy.table import Table

import extinction
from time import time
from gasp.photometry import mag2flux_witherr
import os


def form_filename(root, meta):
    """
    Form file name from meta data
    """

    plate = str(meta["PLATE"])
    mjd = str(meta["MJD"])
    fiber = meta["FIBERID"]
    if fiber < 10:
        fiber = "000" + str(fiber)
    else:
        if fiber < 100:
            fiber = "00" + str(fiber)
        else:
            if fiber < 1000:
                fiber = "0" + str(fiber)
            else:
                fiber = str(fiber)
    # root+'sas/dr12/boss/spectro/redux/v5_7_0/spectra/lite/'+ # old

    return root + "/" + plate + "/spec-" + plate + "-" + mjd + "-" + fiber + ".fits"


def mk_pcaoff_rootname(n_z, ncontam, specsubsampling):
    return (
        "pcaoff_"
        + str(n_z)
        + "_n_components_"
        + str(ncontam)
        + "_ncontam_"
        + str(specsubsampling)
        + "_specsubsampling/"
    )


def load_fluxes(
    fulldata,
    correct_for_extinction=True,
    extinctionCoefficients=[4.239, 3.303, 2.285, 1.698, 1.263],
    bandNames=["u", "g", "r", "i", "z"],
    fieldName="modelMag",
    fieldErrName="modelMagErr",
    fluxesInsteadOfMagnitudes=False,
    mag_zeropoint=22.5,
):
    """
    Load fluxes, remove extinction
    """

    data = np.vstack([fulldata[fieldName + "_" + b] for b in bandNames]).T.astype(float)
    dataErr = np.vstack([fulldata[fieldErrName + "_" + b] for b in bandNames]).T.astype(
        float
    )

    ebv = fulldata["EBV"]
    Av = ebv[:, None] * np.array(extinctionCoefficients)[None, :]
    transmission = 10 ** (-0.4 * Av)

    if fluxesInsteadOfMagnitudes:
        fluxes, fluxErrs = data, dataErr
    else:
        fluxes, fluxErrs = mag2flux_witherr(data, dataErr, mag_zeropoint=mag_zeropoint)

    if correct_for_extinction:
        fluxes /= transmission
        fluxErrs /= transmission

    return fluxes, fluxErrs


def load_DECALS_fluxes(
    fulldata, correct_for_extinction=True, include_wise=False, include_sdss=False
):
    """
    Load fluxes, remove extinction
    """
    # extinctions1 = np.nanmean(
    #    fulldata["GAL_EXT"] / [4.239, 3.303, 2.285, 1.698, 1.263], axis=1
    # )

    # coords = SkyCoord(
    #    ra=np.array(fulldata["RA_1"]),
    #    dec=np.array(fulldata["DEC_1"]),
    #    frame="icrs",
    #    unit="degree",
    # )
    # sfd = SFDQuery()
    # extinctions2 = sfd(coords)

    if correct_for_extinction:
        ebv = fulldata["EBV"]
    else:
        ebv = 0

    bands_decals = ["g", "r", "z"]
    ext = [3.214, 2.165, 1.211]
    if include_wise:
        bands_decals += ["W1", "W2"]
        ext += [0.184, 0.113]

    fluxes_decals = np.vstack([fulldata["FLUX_" + b.upper()] for b in bands_decals]).T
    # trans_decals = np.vstack(
    #    [fulldata["MW_TRANSMISSION_" + b.upper()] for b in bands_decals]
    # ).T
    trans_decals = 10 ** (-0.4 * ebv[:, None] * np.array(ext)[None, :])
    flux_ivars_decals = np.vstack(
        [fulldata["FLUX_IVAR_" + b.upper()] for b in bands_decals]
    ).T
    # fluxes_decals[~np.isfinite(fluxes_decals)] = 0.0
    # if correct_for_extinction:
    fluxes_decals /= trans_decals
    flux_ivars_decals *= trans_decals

    if include_sdss:
        bands_sdss = ["U", "G", "R", "I", "Z"]
        # fluxes_sdss = np.vstack(
        #    [fulldata["PSFFLUX"][:, b] for b in range(len(bands_sdss))]
        # ).T

        trans_sdss = 10 ** (
            -0.4 * ebv[:, None] * np.array([4.239, 3.303, 2.285, 1.698, 1.263])[None, :]
        )
        # trans_sdss = 10 ** (
        #    -0.4 * np.vstack([fulldata["GAL_EXT"][:, b] for b in range(len(bands_sdss))]).T
        # )

        flux_ivars_sdss = np.vstack(
            [fulldata["IVAR_PSFFLUX"][:, b] for b in range(len(bands_sdss))]
        ).T
        fluxes_sdss /= trans_sdss2
        flux_ivars_sdss *= trans_sdss2 ** 2

        bands = (
            ["DECAM_" + b for b in bands_decals[:3]]
            + bands_decals[3:]
            + [b + "_SDSS" for b in bands_sdss]
        )
        fluxes = np.hstack([fluxes_decals, fluxes_sdss])
        flux_ivars = np.hstack([flux_ivars_decals, flux_ivars_sdss])
        bands = np.concatenate([bands_decals, bands_sdss])

        return fluxes, flux_ivars, bands

    else:

        return fluxes_decals, flux_ivars_decals, bands_decals


def make_bounds_from_centers(xcenters):

    xbounds = np.zeros(xcenters.size + 1)
    xbounds[1:-1] = (xcenters[1:] + xcenters[:-1]) / 2
    xbounds[0] = xcenters[0] - (xbounds[1] - xcenters[0])
    xbounds[-1] = xcenters[-1] + (xcenters[-1] - xbounds[-2])

    return xbounds


def build_sdss_grids(zmax):

    # No real need to split the wavelength grid.
    # What matters is that
    # 1. the steps are uniform throughout (for stepped redshifting to work)
    # 2. the steps in redshift correspond to steps in wavelength
    # 3. the wavelength and redshift ranges are sufficient to cover the data of interest.

    step = 0.0001 * 0.98
    # critical for gridsize to be uniform throughout whole grid, .
    logwave_min = 3.55 - np.log10(1 + zmax)
    logwavegrid_z = np.arange(3.55, logwave_min, -step)[::-1]
    logwavegrid_ext = np.arange(3.55, 4.03, step)[1:]
    lamspec_waveoffset = logwavegrid_z.size
    logwavegrid = np.concatenate([logwavegrid_z, logwavegrid_ext])

    print("Size of redshift grid:", logwavegrid_z.size)
    print("Size of wavelength grid:", logwavegrid.size)

    return logwavegrid, logwavegrid_z, lamspec_waveoffset


def loop_nearest(
    indices,
    fulldata,
    root_specdata,
    new_logwave_rest_centers,
    logwavegrid_z,
    length_cap=4800,
    correct_for_extinction=True,
    verbose_step=1000,
    min_valid_pixels=20,
):

    transfer_redshift_grid = 10 ** (logwavegrid_z - logwavegrid_z[0]) - 1

    new_logwave_rest_bounds = make_bounds_from_centers(new_logwave_rest_centers)

    processed_logwave_obs = np.zeros((indices.size, length_cap), dtype=np.float32)
    processed_logwave_obs_start = np.zeros((indices.size,), dtype=np.float32)
    processed_logwave_obs_end = np.zeros((indices.size,), dtype=np.float32)
    processed_redshifts = np.zeros((indices.size,), dtype=np.float32)
    processed_indices = np.zeros((indices.size,), dtype=np.int32)
    processed_index_transfer_redshift = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels_orig = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels = np.zeros((indices.size,), dtype=np.int32)
    processed_index_wave = np.zeros((indices.size,), dtype=np.int32)
    processed_spec = np.zeros((indices.size, length_cap), dtype=np.float32) + np.nan
    processed_spec_off = np.zeros_like(processed_spec) + np.nan
    processed_spec_ivar = np.zeros_like(processed_spec)

    offset = 0
    t1 = time()
    # loop over objects
    for loc, idx in enumerate(indices):

        fname = form_filename(root_specdata, fulldata.iloc[idx])

        if os.path.isfile(fname):
            try:
                data = Table.read(fname)
            except:
                continue
        else:
            continue

        redshift = fulldata.iloc[idx]["Z"]
        if correct_for_extinction:
            ebv = fulldata.iloc[idx]["EBV"]
        else:
            ebv = 0

        logwave_obs, spec, spec_ivar, spec_off = extract_spec(data, ebv=ebv)
        logwave_obs_rest = logwave_obs - np.log10(1 + redshift)

        assert logwave_obs_rest[-1] < new_logwave_rest_centers[-1]

        # Masking sky lines
        ind = np.logical_and(
            logwave_obs >= np.log10(6860), logwave_obs <= np.log10(6920)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7150), logwave_obs <= np.log10(7340)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7575), logwave_obs <= np.log10(7725)
        )
        spec_ivar[ind] = 0
        spec[ind] = np.nan

        #
        diff_log = np.log10(1 + transfer_redshift_grid) - np.log10(1 + redshift)
        index_transfer_redshift = np.argmin(diff_log ** 2.0)
        redshift_ongrid = transfer_redshift_grid[index_transfer_redshift]
        # index_wave = transfer_redshift_grid.size - index_transfer_redshift

        def interpolate(x_target, x_input, test=True):
            x_target_bounds = make_bounds_from_centers(x_target)
            indices = np.searchsorted(x_target_bounds, x_input)
            indices -= 1
            if test:
                assert np.all(x_input >= x_target[0])
                assert np.all(x_input < x_target[-1])
                for i, x in zip(indices, x_input):
                    i2 = np.argmin((x_target - x) ** 2)
                    if i != i2:
                        print(i, i2)
                        print(x_target[i], x_target[i2])
                        print(x_target_bounds[i], x, x_target_bounds[i + 1])
                    assert i == i2
            return indices

        abs_new_indices = interpolate(
            new_logwave_rest_centers, logwave_obs_rest, test=False
        )
        diff_log = logwave_obs_rest[0] - new_logwave_rest_centers
        index_wave2 = np.argmin(diff_log ** 2.0)
        index_wave = abs_new_indices[0]
        assert index_wave == index_wave2
        # print(logwave_obs_rest[0], new_logwave_rest_centers[index_wave], new_logwave_rest_centers[index_wave2])
        # print(new_logwave_rest_centers[index_wave-1], new_logwave_rest_centers[index_wave], new_logwave_rest_centers[index_wave+1])
        # print(index_wave2 - index_wave, end=" ")
        # assert index_wave2 == index_wave

        rel_new_indices = abs_new_indices - index_wave
        if np.any(rel_new_indices < 0) or np.any(rel_new_indices >= length_cap):
            print(
                redshift, index_wave, np.min(rel_new_indices), np.max(rel_new_indices)
            )
            print(logwave_obs[0], logwave_obs.size)
            stop
        assert np.all(rel_new_indices >= 0)
        assert np.all(rel_new_indices < length_cap)

        # any pixels outside of range?
        unique, counts = np.unique(abs_new_indices, return_counts=True)

        if np.any(counts > 1):
            print(idx, redshift)
            print("val", unique[counts > 1][0])
            ind = abs_new_indices == unique[counts > 1][0]
            # print(logwave_obs[ind])
            # print(logwave_obs_rest[ind])
            v = unique[counts > 1][0]
            # print(new_logwave_rest_centers[v-1], new_logwave_rest_centers[v], new_logwave_rest_centers[v+1])
            # print(np.diff(logwave_obs[ind]))
            print(np.sum(counts - 1), end=" ")
        assert np.all(counts <= 1)

        ind = spec_ivar != 0
        ind &= np.isfinite(spec_ivar)
        processed_n_valid_pixels_orig[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)
        processed_n_valid_pixels[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)

        if processed_n_valid_pixels[offset] < min_valid_pixels:
            # print("Skipping", idx)
            continue

        processed_logwave_obs_start[offset] = logwave_obs[0]
        processed_logwave_obs_end[offset] = logwave_obs[-1]
        processed_logwave_obs[offset, :][rel_new_indices] = logwave_obs
        processed_indices[offset] = idx

        processed_spec[offset, :][rel_new_indices] = spec
        processed_spec_ivar[offset, :][rel_new_indices] = spec_ivar
        processed_spec_off[offset, :][rel_new_indices] = spec_off

        processed_redshifts[offset] = redshift
        processed_index_transfer_redshift[offset] = index_transfer_redshift
        processed_index_wave[offset] = index_wave

        offset += 1

        if loc > 0 and loc % verbose_step == 0:
            t2 = time()
            print(
                "Processed",
                loc,
                "spectra in %.2f" % ((t2 - t1) / 60),
                "minutes (%.3f" % ((t2 - t1) / loc),
                "sec per object)",
            )
            print("Valid spectra:", offset, "out of", loc)
            t3 = (indices.size - loc) * ((t2 - t1) / loc)
            print(
                "> Estimated remaining time: >> %.2f" % (t3 / 60.0),
                "minutes << for",
                indices.size - loc,
                "objects",
            )

    chi2s_off = np.nansum(
        (processed_spec[:offset, :] - processed_spec_off[:offset, :]) ** 2
        * processed_spec_ivar[:offset, :],
        axis=-1,
    )

    return (
        transfer_redshift_grid,
        processed_indices[:offset],
        processed_logwave_obs[:offset, :],
        processed_logwave_obs_start[:offset],
        processed_logwave_obs_end[:offset],
        processed_redshifts[:offset],
        processed_spec[:offset, :],
        processed_spec_ivar[:offset, :],
        processed_spec_off[:offset, :],
        processed_index_transfer_redshift[:offset],
        processed_index_wave[:offset],
        processed_n_valid_pixels[:offset],
        processed_n_valid_pixels_orig[:offset],
        chi2s_off,
    )


def rebin_nearest_indices(
    new_logwave_rest_bounds,
    redshift,
    transfer_redshift_grid,
    logwave_obs_middle,
    length_cap=None,
):

    object_logwave_rest = logwave_obs_middle - np.log10(1 + redshift)

    # positions of data in new pixels
    x_new_indices = np.searchsorted(
        logwave_obs_middle, new_logwave_rest_bounds, side="right"
    )

    # exclude redundant values at beginning or end:
    # correct = np.logical_and(
    #    x_new_indices != x_new_indices[0], x_new_indices != x_new_indices[-1]
    # )
    # x_new_indices = x_new_indices[correct]

    if length_cap is not None:
        x_new_indices = x_new_indices[0:length_cap]

    diff_log = np.log10(1 + transfer_redshift_grid) - np.log10(1 + redshift)
    index_transfer_redshift = np.argmin(diff_log ** 2.0)

    return x_new_indices, index_transfer_redshift


def extract_spec(data, ebv=0, RV=3.1):

    mask = data["AND_MASK"].data > 0
    logwave_obs = data["LOGLAM"].astype(float).data
    spec = data["FLUX"].astype(float).data
    spec[mask] = np.nan
    spec_ivar = data["IVAR"].astype(float).data
    spec_ivar[mask] = np.nan
    spec_off = data["MODEL"].astype(float).data
    # spec = np.ma.masked_array(data["FLUX"].astype(float).data, mask=mask)
    # spec_off = np.ma.masked_array(data["MODEL"].astype(float).data, mask=mask)
    # spec_ivar = np.ma.masked_array(data["IVAR"].astype(float).data, mask=mask)

    if ebv > 0:
        extinction_sed = ebv * extinction.ccm89(10 ** logwave_obs, 1.0, RV, unit="aa")
        spec = extinction.remove(extinction_sed, spec)
        spec_off = extinction.remove(extinction_sed, spec_off)
        spec_ivar = extinction.apply(extinction_sed, spec_ivar)

    # ind = np.isfinite(logwave_obs)
    # ind &= np.isfinite(spec)
    # ind &= np.isfinite(spec_ivar)
    # ind &= np.isfinite(spec_off)

    return logwave_obs, spec, spec_ivar, spec_off


def save_spectrophotometry(
    root,
    processed_redshifts,
    lamspec_waveoffset,
    processed_index_wave,
    processed_index_transfer_redshift,
    wave_rest_interp,
    lam_phot_eff,
    lam_phot_size_eff,
    transfermatrix_phot,
    transfer_redshift_grid,
    processed_spec,
    processed_spec_off,
    processed_spec_ivar,
    chi2s_sdss,
    fluxes,
    flux_ivars,
    processed_interprightindices,
    processed_interpweights,
):

    np.save(root + "chi2s_sdss", chi2s_sdss)
    # np.save(root + "valid_ids", processed_indices)
    np.save(root + "redshifts", processed_redshifts)
    np.save(root + "lamspec_waveoffset", lamspec_waveoffset)
    np.save(root + "interprightindices", processed_interprightindices)
    np.save(root + "interpweights", processed_interpweights)
    np.save(root + "index_wave", processed_index_wave)
    np.save(root + "index_transfer_redshift", processed_index_transfer_redshift)
    np.save(root + "lamgrid", wave_rest_interp)
    np.save(root + "lam_phot_eff", lam_phot_eff)
    np.save(root + "lam_phot_size_eff", lam_phot_size_eff)
    np.save(root + "transferfunctions", transfermatrix_phot)
    np.save(root + "transferfunctions_zgrid", transfer_redshift_grid)
    print("saved transferfunctions")
    np.save(root + "spec", processed_spec)
    np.save(root + "spec_mod", processed_spec_off)
    np.save(root + "spec_invvar", processed_spec_ivar)
    print("saved seds")
    np.save(root + "phot", fluxes)
    np.save(root + "phot_invvar", flux_ivars)
    print("saved photometry")
    print("all done")


def loop_interp(
    indices,
    fulldata,
    root_specdata,
    new_logwave_rest_centers,
    logwavegrid_z,
    length_cap=4800,
    correct_for_extinction=True,
    verbose_step=1000,
    min_valid_pixels=20,
):

    transfer_redshift_grid = 10 ** (logwavegrid_z - logwavegrid_z[0]) - 1

    new_logwave_rest_bounds = make_bounds_from_centers(new_logwave_rest_centers)

    processed_logwave_obs = np.zeros((indices.size, length_cap), dtype=np.float32)
    processed_logwave_obs_start = np.zeros((indices.size,), dtype=np.float32)
    processed_logwave_obs_end = np.zeros((indices.size,), dtype=np.float32)
    processed_redshifts = np.zeros((indices.size,), dtype=np.float32)
    processed_indices = np.zeros((indices.size,), dtype=np.int32)
    processed_index_transfer_redshift = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels_orig = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels = np.zeros((indices.size,), dtype=np.int32)
    processed_index_wave = np.zeros((indices.size,), dtype=np.int32)
    processed_spec = np.zeros((indices.size, length_cap), dtype=np.float32) + np.nan
    processed_spec_off = np.zeros_like(processed_spec) + np.nan
    processed_spec_ivar = np.zeros_like(processed_spec)

    offset = 0
    t1 = time()
    # loop over objects
    for loc, idx in enumerate(indices):

        fname = form_filename(root_specdata, fulldata.iloc[idx])

        if os.path.isfile(fname):
            try:
                data = Table.read(fname)
            except:
                continue
        else:
            continue

        redshift = fulldata.iloc[idx]["Z"]
        if correct_for_extinction:
            ebv = fulldata.iloc[idx]["EBV"]
        else:
            ebv = 0

        logwave_obs, spec, spec_ivar, spec_off = extract_spec(data, ebv=ebv)
        logwave_obs_rest = logwave_obs - np.log10(1 + redshift)

        assert logwave_obs_rest[-1] < new_logwave_rest_centers[-1]

        # Masking sky lines
        ind = np.logical_and(
            logwave_obs >= np.log10(6860), logwave_obs <= np.log10(6920)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7150), logwave_obs <= np.log10(7340)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7575), logwave_obs <= np.log10(7725)
        )
        spec_ivar[ind] = 0
        spec[ind] = np.nan

        #
        diff_log = np.log10(1 + transfer_redshift_grid) - np.log10(1 + redshift)
        index_transfer_redshift = np.argmin(diff_log ** 2.0)
        redshift_ongrid = transfer_redshift_grid[index_transfer_redshift]

        diff_log = logwave_obs_rest[0] - new_logwave_rest_centers
        index_wave = np.argmin(diff_log ** 2.0)

        def interp(y):
            return scipy.interpolate.interp1d(
                logwave_obs_rest,
                y,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )(new_logwave_rest_centers[index_wave : index_wave + length_cap])

        spec_interp = interp(spec)
        spec_ivar_interp = interp(spec_ivar)
        spec_off_interp = interp(spec_off)
        logwave_obs_interp = interp(logwave_obs)

        ind = spec_ivar != 0
        ind &= np.isfinite(spec_ivar)
        processed_n_valid_pixels_orig[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)
        processed_n_valid_pixels[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)

        if processed_n_valid_pixels[offset] < min_valid_pixels:
            # print("Skipping", idx)
            continue

        processed_logwave_obs_start[offset] = logwave_obs[0]
        processed_logwave_obs_end[offset] = logwave_obs[-1]
        processed_logwave_obs[offset, :] = logwave_obs_interp
        processed_indices[offset] = idx

        processed_spec[offset, :] = spec_interp
        processed_spec_ivar[offset, :] = spec_ivar_interp
        processed_spec_off[offset, :] = spec_off_interp

        processed_redshifts[offset] = redshift
        processed_index_transfer_redshift[offset] = index_transfer_redshift
        processed_index_wave[offset] = index_wave

        offset += 1

        if loc > 0 and loc % verbose_step == 0:
            t2 = time()
            print(
                "Processed",
                loc,
                "spectra in %.2f" % ((t2 - t1) / 60),
                "minutes (%.3f" % ((t2 - t1) / loc),
                "sec per object)",
            )
            print("Valid spectra:", offset, "out of", loc)
            t3 = (indices.size - loc) * ((t2 - t1) / loc)
            print(
                "> Estimated remaining time: >> %.2f" % (t3 / 60.0),
                "minutes << for",
                indices.size - loc,
                "objects",
            )

    chi2s_off = np.nansum(
        (processed_spec[:offset, :] - processed_spec_off[:offset, :]) ** 2
        * processed_spec_ivar[:offset, :],
        axis=-1,
    )

    return (
        transfer_redshift_grid,
        processed_indices[:offset],
        processed_logwave_obs[:offset, :],
        processed_logwave_obs_start[:offset],
        processed_logwave_obs_end[:offset],
        processed_redshifts[:offset],
        processed_spec[:offset, :],
        processed_spec_ivar[:offset, :],
        processed_spec_off[:offset, :],
        processed_index_transfer_redshift[:offset],
        processed_index_wave[:offset],
        processed_n_valid_pixels[:offset],
        processed_n_valid_pixels_orig[:offset],
        chi2s_off,
    )


def loop_interptransfer(
    indices,
    fulldata,
    root_specdata,
    new_logwave_rest_centers,
    logwavegrid_z,
    length_cap=4800,
    correct_for_extinction=True,
    verbose_step=1000,
    min_valid_pixels=20,
):

    transfer_redshift_grid = 10 ** (logwavegrid_z - logwavegrid_z[0]) - 1

    new_logwave_rest_bounds = make_bounds_from_centers(new_logwave_rest_centers)

    processed_logwave_obs = np.zeros((indices.size, length_cap), dtype=np.float32)
    processed_interprightindices = np.zeros((indices.size, length_cap), dtype=np.int32)
    processed_interpweights = np.zeros((indices.size, length_cap), dtype=np.float32)
    processed_redshifts = np.zeros((indices.size,), dtype=np.float32)
    processed_indices = np.zeros((indices.size,), dtype=np.int32)
    processed_index_transfer_redshift = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels_orig = np.zeros((indices.size,), dtype=np.int32)
    processed_n_valid_pixels = np.zeros((indices.size,), dtype=np.int32)
    processed_index_wave = np.zeros((indices.size,), dtype=np.int32)
    processed_spec = np.zeros((indices.size, length_cap), dtype=np.float32) + np.nan
    processed_spec_off = np.zeros_like(processed_spec) + np.nan
    processed_spec_ivar = np.zeros_like(processed_spec)

    offset = 0
    t1 = time()
    # loop over objects
    for loc, idx in enumerate(indices):

        fname = form_filename(root_specdata, fulldata.iloc[idx])

        if os.path.isfile(fname):
            try:
                data = Table.read(fname)
            except:
                print("Error loading file", idx, fname)
                continue
        else:
            print("File not found:", idx, fname)
            continue

        redshift = fulldata.iloc[idx]["Z"]
        if correct_for_extinction:
            ebv = fulldata.iloc[idx]["EBV"]
        else:
            ebv = 0

        logwave_obs, spec, spec_ivar, spec_off = extract_spec(data, ebv=ebv)
        logwave_obs_rest = logwave_obs - np.log10(1 + redshift)

        assert logwave_obs_rest[-1] < new_logwave_rest_centers[-1]

        # Masking sky lines
        ind = np.logical_and(
            logwave_obs >= np.log10(6860), logwave_obs <= np.log10(6920)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7150), logwave_obs <= np.log10(7340)
        )
        ind |= np.logical_and(
            logwave_obs >= np.log10(7575), logwave_obs <= np.log10(7725)
        )
        spec_ivar[ind] = 0
        spec[ind] = np.nan

        #
        diff_log = np.log10(1 + transfer_redshift_grid) - np.log10(1 + redshift)
        index_transfer_redshift = np.argmin(diff_log ** 2.0)
        redshift_ongrid = transfer_redshift_grid[index_transfer_redshift]

        diff_log = logwave_obs_rest[0] - new_logwave_rest_centers
        index_wave = np.argmin(diff_log ** 2.0)

        interprightindices, interpweights = interp_coefficients(
            new_logwave_rest_centers, logwave_obs_rest
        )

        ind = spec_ivar != 0
        ind &= np.isfinite(spec_ivar)
        processed_n_valid_pixels_orig[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)
        processed_n_valid_pixels[offset] = np.sum(ind)  # np.sum(~spec_ivar.mask)

        if processed_n_valid_pixels[offset] < min_valid_pixels:
            # print("Skipping", idx)
            print(
                "Too few pixels for file:", idx, fname, processed_n_valid_pixels[offset]
            )
            continue

        sz = spec.size

        if True:
            processed_interprightindices[offset, :sz] = interprightindices
            processed_interpweights[offset, :sz] = interpweights
            processed_logwave_obs[offset, :sz] = logwave_obs
            processed_spec[offset, :sz] = spec
            processed_spec_ivar[offset, :sz] = spec_ivar
            processed_spec_off[offset, :sz] = spec_off
        else:
            indloc = interprightindices - index_wave - 1
            processed_interprightindices[offset, :] = np.arange(length_cap)
            processed_interpweights[offset, :][indloc] = interpweights
            processed_logwave_obs[offset, :][indloc] = logwave_obs
            processed_spec[offset, :][indloc] = spec
            processed_spec_ivar[offset, :][indloc] = spec_ivar
            processed_spec_off[offset, :][indloc] = spec_off

        processed_indices[offset] = idx
        processed_redshifts[offset] = redshift
        processed_index_transfer_redshift[offset] = index_transfer_redshift
        processed_index_wave[offset] = index_wave

        offset += 1

        if loc > 0 and loc % verbose_step == 0:
            t2 = time()
            print(
                "Processed",
                loc,
                "spectra in %.2f" % ((t2 - t1) / 60),
                "minutes (%.3f" % ((t2 - t1) / loc),
                "sec per object)",
            )
            print("Valid spectra:", offset, "out of", loc)
            t3 = (indices.size - loc) * ((t2 - t1) / loc)
            print(
                "> Estimated remaining time: >> %.2f" % (t3 / 60.0),
                "minutes << for",
                indices.size - loc,
                "objects",
            )

    chi2s_off = np.nansum(
        (processed_spec[:offset, :] - processed_spec_off[:offset, :]) ** 2
        * processed_spec_ivar[:offset, :],
        axis=-1,
    )

    return (
        transfer_redshift_grid,
        processed_indices[:offset],
        processed_logwave_obs[:offset, :],
        processed_interprightindices[:offset],
        processed_interpweights[:offset],
        processed_redshifts[:offset],
        processed_spec[:offset, :],
        processed_spec_ivar[:offset, :],
        processed_spec_off[:offset, :],
        processed_index_transfer_redshift[:offset],
        processed_index_wave[:offset],
        processed_n_valid_pixels[:offset],
        processed_n_valid_pixels_orig[:offset],
        chi2s_off,
    )
