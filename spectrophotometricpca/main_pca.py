import os
import time
import re
import numpy as onp
import click
import itertools

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


if True:
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

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")
print("CPUs and GPUs:", cpus, gpus)
jax.config.update("jax_platform_name", "cpu")
print("Devices in use:", jax.devices(), jax.device_count())

import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random
import jax.experimental.optimizers

# Global flag to set a specific platform, must be used at startup.

key = random.PRNGKey(42)

from datapipeline import *
from pca import *


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
    default=101,
    show_default=True,
    type=int,
    help="Number of epochs",
)
@click.option(
    "--batchsize",
    default=512,
    show_default=True,
    type=int,
    help="Batch size for training",
)
@click.option(
    "--n_components",
    default=4,
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
    default=1e-3,
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
    default=False,
    show_default=True,
    type=bool,
    help="Only using spectroscopy, as opposed to both photometry and spectroscopy",
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
):
    output_valid_steps = 2
    use_subset = False
    write_subset = False

    output_dir = results_dir + "/" + dataname + "/"
    output_dir += pca_file_prefix(
        n_components, n_poly, batchsize, subsampling, learningrate
    )
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
    indices_train, indices_valid = datapipe.indices[0::2], datapipe.indices[1::2]
    print("Size of training:", indices_train.size)
    print("Size of validation:", indices_valid.size)

    n_obj_train, valid_n_obj = indices_train.size, indices_valid.size
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
        lamgrid_spec,
    ) = datapipe.get_grids()

    np.save(output_dir + "/train_indices", indices_train)
    np.save(output_dir + "/valid_indices", indices_valid)

    # output chi2s of SDSS models
    # np.save(prefix + "valid_chi2s_sdss", self.chi2s_sdss[datapipe.ind_valid_orig])

    # parameters:  creating the priors, functions of redshift
    # Initial valuesn_components, n_poly, lamgridsize
    polynomials_spec = chebychevPolynomials(n_poly, n_pix_spec)

    pcamodel = PCAModel(polynomials_spec, output_prefix, "")
    params, pcacomponents_prior = pcamodel.init_params(
        key, n_components, n_poly, n_pix_sed
    )

    # Initialise optimiser, as well as update operation.
    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(learningrate)
    opt_state = opt_init(params)

    if speconly:

        suffix = "_speconly"

        bayesianpca = jit(
            pcamodel.bayesianpca_speconly, backend="gpu", static_argnums=(1, 2)
        )

        @partial(jit, static_argnums=(1, 2), backend="gpu")
        def loss_fn(params, data_batch, polynomials_spec):
            return pcamodel.loss_speconly(params, data_batch, polynomials_spec)

    else:

        suffix = "_specandphot"

        bayesianpca = jit(
            pcamodel.bayesianpca_specandphot, backend="gpu", static_argnums=(1, 2)
        )

        @partial(jit, static_argnums=(1, 2), backend="gpu")
        def loss_fn(params, data_batch, polynomials_spec):
            return pcamodel.loss_specandphot(params, data_batch, polynomials_spec)

    @partial(jit, static_argnums=(2, 3), backend="gpu")
    def update(step, opt_state, data_batch, data_aux):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_fn)(params, data_batch, data_aux)
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    # Start loop
    losses_train = onp.zeros((n_epochs, numBatches_train))
    losses_valid = onp.zeros((n_epochs, numBatches_valid))
    itercount = itertools.count()
    previous_validation_loss1, previous_validation_loss2 = np.inf, np.inf
    start_time = time.time()
    print("Starting training")
    i_start = 0
    for i in range(i_start, n_epochs):

        neworder = jax.random.permutation(key, np.arange(numBatches_train))
        indices_train_reordered = np.take(indices_train, neworder)
        datapipe.batch = 0  # reset batch number
        for j in range(numBatches_train):
            data_batch = datapipe.next_batch(indices_train_reordered, batchsize)
            the_loss, opt_state = update(
                next(itercount), opt_state, data_batch, polynomials_spec
            )
            losses_train[i, j] = the_loss.block_until_ready()
            # will force all outputs of jit fct

            # batch_ells = train_specphotscaling[neworder[si : si + bs]]
            # update scaling, calculated from spec only
            # updated_batch_ells[updated_batch_ells < 0] = 1.0
            # train_specphotscaling[neworder[si : si + bs]] = updated_batch_ells

        loss = np.mean(losses_train[i, :])

        print(
            "> Training loss: %.5e" % loss,
            end=" - ",
        )
        print(
            "Elapsed time: %dh %dm %ds" % process_time(start_time, time.time()),
            end=" - ",
        )
        print(
            "Remaining time: %dh %dm %ds"
            % process_time(
                start_time,
                time.time(),
                multiply=(n_epochs - i) / float(i + 1 - i_start),
            ),
            end="\n",
        )

        pcamodel.write_model()

        if i % output_valid_steps == 0:

            print("> Running validation models and data")
            resultspipe = ResultsPipeline(
                output_prefix,
                suffix,
                n_components + n_poly,
                datapipe,
                indices_valid,
            )

            datapipe.batch = 0  # reset batch number
            for j in range(numBatches_valid):
                data_batch = datapipe.next_batch(indices_valid, batchsize)

                result = bayesianpca(params, data_batch, polynomials_spec)
                losses_valid[i, j] = -np.sum(result[0].block_until_ready())

                resultspipe.write_batch(data_batch, *result)

            current_validation_loss = np.mean(losses_valid[i, :])

            resultspipe.write_reconstructions()
            del resultspipe

            print("> Validation loss: %.5e" % current_validation_loss)
            if (
                current_validation_loss > previous_validation_loss1
                and current_validation_loss > previous_validation_loss2
            ):
                print(
                    "Current validation loss if worse than the previous two - early stopping"
                )
                exit(0)

            print("> Back to training")
            previous_validation_loss2 = previous_validation_loss1
            previous_validation_loss1 = current_validation_loss

            np.save(output_dir + "/valid_losses", losses_valid[: i + 1, :])

    print("Learning finished")


if __name__ == "__main__":
    main()
