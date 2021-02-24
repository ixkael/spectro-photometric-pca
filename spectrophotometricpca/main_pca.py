import os
import time
import re
import numpy as onp
import click
import itertools

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # no reason not to preallocate!


if False:
    # Force multiple CPU devices... but not that useful since no CPU-parallel operations?

    xla_flags = os.getenv("XLA_FLAGS", "").lstrip("--")
    xla_flags = re.sub(
        r"xla_force_host_platform_device_count=.+\s", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(16)] + xla_flags
    )
    os.environ.update(
        XLA_FLAGS=(
            "--xla_cpu_multi_thread_eigen=true "
            "--xla_force_host_platform_device_count=16 "
            "intra_op_parallelism_threads=8 "
            "inter_op_parallelism_threads=2 "
        ),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )

import jax
from jax.interpreters import xla

# jax.core.check_leaks = True

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")
print("CPUs and GPUs:", cpus, gpus)
jax.config.update("jax_platform_name", "cpu")  # will allocate to GPU where relevant
print("Devices in use:", jax.devices(), jax.device_count())

import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random
import jax.experimental.optimizers

# Global flag to set a specific platform, must be used at startup.

key = random.PRNGKey(42)

from datapipeline import *
from pca import *
from utils import *


@click.command()
@click.argument("input_dir", type=click.Path(exists=True), default="data/dr16eboss")
# @click.option("--input_dir", required=True, type=str, help="Root directory of data set")
@click.option(
    "--dataname",
    default=None,
    callback=lambda c, p, v: v if v else c.params["input_dir"].split("/")[-1],
    help="Number of epochs",
)
@click.option(
    "--results_dir",
    default="results/",
    type=click.Path(exists=True),
    show_default=True,
    help="Directory for outputs",
)
@click.option(
    "--n_epochs",
    default=200,
    show_default=True,
    type=int,
    help="Number of epochs",
)
@click.option(
    "--batchsize",
    default=3000,  # * 2 or 8,
    show_default=True,
    type=int,
    help="Batch size for training",
)
@click.option(
    "--n_components",
    default=12,
    show_default=True,
    type=int,
    help="Number of PCA components",
)
@click.option(
    "--subsampling",
    default=1,
    show_default=True,
    type=int,
    help="Subsampling of spectra and SEDs",
)
@click.option(
    "--learningrate",
    default=1e-2,
    show_default=True,
    type=float,
    help="Learning rate for optimisation",
)
@click.option(
    "--n_poly",
    default=3,
    show_default=True,
    type=int,
    help="Number of chebychev polynomials for spectroscopic systematics",
)
@click.option(
    "--speconly",
    default=True,
    show_default=True,
    type=bool,
    help="Only using spectroscopy, as opposed to both photometry and spectroscopy",
)
@click.option(
    "--opt_basis",
    default=False,
    show_default=True,
    type=bool,
    help="Optimize basis functions",
)
@click.option(
    "--opt_priors",
    default=True,
    show_default=True,
    type=bool,
    help="Optimize priors",
)
# flag.DEFINE_boolean("load", False, "Loading existing model")
# flag.DEFINE_integer("alldata", 0, "")
# flag.DEFINE_integer("computeofficialredshiftposteriors", 0, "")
def main(
    input_dir,
    dataname,
    results_dir,
    n_epochs,
    batchsize,
    n_components,
    subsampling,
    learningrate,
    n_poly,
    speconly,
    opt_basis,
    opt_priors,
):
    output_valid_zsteps = n_epochs - 1
    compute_redshifts_pdfs = True
    early_stopping = False
    use_subset = False
    write_subset = False

    output_dir = results_dir + "/" + dataname + "/"
    the_prefix = pca_file_prefix(
        n_components,
        n_poly,
        batchsize,
        subsampling,
        opt_basis,
        opt_priors,
        learningrate,
    )
    print(the_prefix)
    output_dir += the_prefix
    output_dir += "/"
    output_prefix = output_dir + ""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("prefix : \n", output_prefix)

    datapipe = DataPipeline(
        input_dir=input_dir + "/",
        subsampling=subsampling,
        use_subset=use_subset,
        write_subset=write_subset,
    )

    # Prepare copies of training indices
    # indices_train, indices_valid = datapipe.indices[0::2], datapipe.indices[1::2] # half-half
    indices_train = onp.load(input_dir + "/indices_train.npy")
    indices_valid = onp.load(input_dir + "/indices_valid.npy")
    print("Size of training before cuts:", indices_train.size)
    indices_train = indices_train[np.in1d(indices_train, datapipe.indices)]
    print("Size of training after cuts: ", indices_train.size)
    print("Size of validation before cuts:", indices_valid.size)
    indices_valid = indices_valid[np.in1d(indices_valid, datapipe.indices)]
    print("Size of validation after cuts:", indices_valid.size)
    onp.save(output_dir + "/indices_train", indices_train)
    onp.save(output_dir + "/indices_valid", indices_valid)

    numObj_train, numObj_valid = indices_train.size, indices_valid.size
    numBatches_train = datapipe.get_nbatches(indices_train, batchsize)
    numBatches_valid = datapipe.get_nbatches(indices_valid, batchsize)

    # Extract grids and various useful numbers
    (
        lamgrid,
        lam_phot_eff,
        lam_phot_size_eff,
        transferfunctions,
        transferfunctions_zgrid,
        n_pix_sed,
        n_pix_spec,
        numBands,
    ) = datapipe.get_grids()

    # output chi2s of SDSS models
    # np.save(prefix + "valid_chi2s_sdss", self.chi2s_sdss[datapipe.ind_valid_orig])

    # parameters:  creating the priors, functions of redshift
    # Initial valuesn_components, n_poly, n_pix_sed
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)

    pcamodel = PCAModel(polynomials_spec, output_prefix, "")
    pcacomponents_prior = pcamodel.init_params(
        key, n_components, n_poly, n_pix_sed, opt_basis, opt_priors
    )

    # Load PCA templates
    pcamodel.pcacomponents = load_fits_templates(lamgrid, n_components)

    print("n_components, n_pix_spec", n_components, n_pix_spec)

    # Initialise optimiser, as well as update operation.
    opt_init, opt_update, get_params_opt = jax.experimental.optimizers.adam(
        learningrate
    )
    opt_state = opt_init(pcamodel.get_params_opt())

    if speconly:
        suffix_orig = "_speconly"
        bayesianpca = jit(
            bayesianpca_speconly, backend="gpu", static_argnums=(3, 4, 5, 6)
        )
        loss_fn = jit(loss_speconly, backend="gpu", static_argnums=(3, 4, 5, 6))
    else:
        suffix_orig = "_specandphot"
        bayesianpca = jit(
            bayesianpca_specandphot, backend="gpu", static_argnums=(3, 4, 5, 6)
        )
        loss_fn = jit(loss_specandphot, backend="gpu", static_argnums=(3, 4, 5, 6))

    if opt_basis and opt_priors:
        data_aux = polynomials_spec
    if opt_basis and not opt_priors:
        priors = pcamodel.get_params_nonopt()
        data_aux = (priors, polynomials_spec)
    if not opt_basis and opt_priors:
        components = pcamodel.get_params_nonopt()[0]
        data_aux = (components, polynomials_spec)

    @partial(jit, backend="gpu", static_argnums=(4, 5, 6, 7))
    def update(
        step,
        opt_state,
        data_batch,
        data_aux,
        n_components,
        n_pix_spec,
        opt_basis,
        opt_priors,
    ):
        params = get_params_opt(opt_state)
        value, grads = jax.value_and_grad(loss_fn)(
            params,
            data_batch,
            data_aux,
            n_components,
            n_pix_spec,
            opt_basis,
            opt_priors,
        )
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    # Start loop
    losses_train = onp.zeros((n_epochs, numBatches_train))
    losses_valid = onp.zeros((n_epochs, numBatches_valid))
    itercount = itertools.count()
    previous_validation_loss1, previous_validation_loss2 = onp.inf, onp.inf
    start_time = time.time()
    print("Starting training")
    i_start = 0
    for i in range(i_start, n_epochs):

        if i % output_valid_zsteps == 0 and i > 0:

            print("> Running validation models and data")
            if i == 0:
                suffix = suffix_orig + "_init"
            else:
                suffix = suffix_orig
            resultspipe = ResultsPipeline(
                output_prefix,
                suffix,
                n_components + n_poly,
                datapipe,
                indices_valid,
            )

            datapipe.batch = 0  # reset batch number
            valid_start_time = time.time()
            for j in range(numBatches_valid):
                data_batch = datapipe.next_batch(indices_valid, batchsize)

                result = bayesianpca(
                    pcamodel.get_params_opt(),
                    data_batch,
                    data_aux,
                    n_components,
                    n_pix_spec,
                    opt_basis,
                    opt_priors,
                )
                losses_valid[i, j] = -np.sum(result[0].block_until_ready())

                resultspipe.write_batch(data_batch, *result)
                del result

            valid_end_time = time.time()
            current_validation_loss = onp.mean(losses_valid[i, :])

            resultspipe.write_reconstructions()
            del resultspipe
            xla._xla_callable.cache_clear()

            print("> Validation loss: %.7e" % current_validation_loss)
            if (
                current_validation_loss > previous_validation_loss1
                and current_validation_loss > previous_validation_loss2
            ):
                print("Validation loss is worse than the previous two")
                if early_stopping:
                    print("Early stopping")
                    exit(0)

            if compute_redshifts_pdfs:
                print("> Running redshift grids")
                zstep = 1
                print(
                    "> should take approximately %dh %dm %ds"
                    % process_time(
                        valid_start_time,
                        valid_end_time,
                        transferfunctions_zgrid[::zstep].size,
                    )
                )
                valid_logpz = (
                    onp.zeros((numObj_valid, transferfunctions_zgrid[::zstep].size))
                    + onp.nan
                )
                valid_thetamap_z = (
                    onp.zeros(
                        (
                            numObj_valid,
                            transferfunctions_zgrid[::zstep].size,
                            n_components + n_poly,
                        )
                    )
                    + onp.nan
                )
                datapipe.batch = 0  # reset batch number

                onp.save(
                    output_dir + "/logpz_zgrid" + suffix,
                    transferfunctions_zgrid[::zstep],
                )

                for j in range(numBatches_valid):
                    data_batch = datapipe.next_batch(indices_valid, batchsize)
                    si, bs = data_batch[0], data_batch[1]

                    print("> Batch", j + 1, "out of", numBatches_valid)
                    batch_start_time = time.time()

                    for iz, z in enumerate(transferfunctions_zgrid[::zstep]):

                        if iz < 2:
                            # currently numerically unstable due to batch_transferfunctions.
                            # will need to investigate why at some point.
                            continue

                        result = bayesianpca(
                            pcamodel.get_params_opt(),
                            datapipe.change_redshift(iz, zstep, data_batch),
                            data_aux,
                            n_components,
                            n_pix_spec,
                            opt_basis,
                            opt_priors,
                        )
                        valid_logpz[si : si + bs, iz] = result[0].block_until_ready()
                        valid_thetamap_z[si : si + bs, iz, :] = result[
                            1
                        ].block_until_ready()
                        xla._xla_callable.cache_clear()

                    onp.save(
                        output_dir + "/logpz" + suffix,
                        valid_logpz[: si + bs, :],
                    )
                    onp.save(
                        output_dir + "/thetamap_z" + suffix,
                        valid_thetamap_z[: si + bs, :, :],
                    )

                    print_remaining_time(batch_start_time, 0, j, numBatches_valid)

            print_elapsed_time(start_time)

            previous_validation_loss2 = previous_validation_loss1
            previous_validation_loss1 = current_validation_loss

            onp.save(output_dir + "/valid_losses" + suffix, losses_valid[: i + 1, :])
            print("> Back to training")
            xla._xla_callable.cache_clear()

        neworder = jax.random.permutation(key, np.arange(numObj_train))
        indices_train_reordered = np.take(indices_train, neworder)
        datapipe.batch = 0  # reset batch number
        for j in range(numBatches_train):
            data_batch = datapipe.next_batch(indices_train_reordered, batchsize)
            the_loss, opt_state = update(
                next(itercount),
                opt_state,
                data_batch,
                data_aux,
                n_components,
                n_pix_spec,
                opt_basis,
                opt_priors,
            )
            # logfml, thetamap, thetastd, specmod_map, photmod_map = bayesianpca(
            #    params, data_batch, polynomials_spec
            # )
            losses_train[i, j] = the_loss.block_until_ready()
            # will force all outputs of jit fct

            # batch_ells = train_specphotscaling[neworder[si : si + bs]]
            # update scaling, calculated from spec only
            # updated_batch_ells[updated_batch_ells < 0] = 1.0
            # train_specphotscaling[neworder[si : si + bs]] = updated_batch_ells

        loss = onp.mean(losses_train[i, :])
        pcamodel.set_params(get_params_opt(opt_state))  # get updated parameter array
        xla._xla_callable.cache_clear()

        print(
            "> Training loss: %.7e" % loss,
            " (iteration " + str(i) + ")",
            end=" - ",
        )
        print_elapsed_time(start_time)

        # write model to file!
        pcamodel.write_model()

    print("Learning finished")


if __name__ == "__main__":
    main()
