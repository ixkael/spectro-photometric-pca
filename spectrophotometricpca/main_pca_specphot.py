import os
import time
import numpy as onp
import click
import itertools

os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit --xla_gpu_cuda_data_dir"

import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random
import jax.experimental.optimizers

# Global flag to set a specific platform, must be used at startup.
jax.config.update("jax_platform_name", "cpu")
print(jax.devices("cpu"))
# print(jax.devices("gpu"))

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
    default="/Users/bl/spectro-photometric-encoder-results/",
    type=click.Path(exists=True),
    show_default=True,
    help="Directory for outputs",
)
@click.option(
    "--n_epochs",
    default=1,
    show_default=True,
    type=int,
    help="Number of epochs",
)
@click.option(
    "--batchsize",
    default=256,
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
    default=16,
    show_default=True,
    type=int,
    help="Subsampling of spectra and SEDs",
)
@click.option(
    "--n_threads",
    default=-1,
    show_default=True,
    type=int,
    help="Number of threads (-1 is GPU)",
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
    n_threads,
    learningrate,
    n_poly,
):
    output_valid_steps = 2
    output_valid_redshift_steps = 100
    nplots = 6
    use_subset = False
    write_subset = False

    if n_threads > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Number of threads:", n_threads)

    output_dir = results_dir + dataname + "/"
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

    pcamodel_speconly = PCAModel(polynomials_spec, output_prefix, "_speconly")
    pcamodel_specandphot = PCAModel(polynomials_spec, output_prefix, "_specandphot")
    params_speconly, pcacomponents_prior_speconly = pcamodel_speconly.init_params(
        key, n_components, n_poly, n_pix_sed
    )
    (
        params_specandphot,
        pcacomponents_prior_specandphot,
    ) = pcamodel_specandphot.init_params(key, n_components, n_poly, n_pix_sed)
    # prepare spec calibration polynomials
    params = [params_speconly, params_specandphot]

    # Initialise optimiser, as well as update operation.
    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(learningrate)
    opt_state = opt_init(params)

    @partial(jit, static_argnums=(1, 2))
    def loss_spec_and_specandphot(params_all, data_batch, polynomials_spec):
        [params_speconly, params_specandphot] = params_all
        return pcamodel_speconly.loss_speconly(
            params_speconly, data_batch, polynomials_spec
        ) + pcamodel_specandphot.loss_specandphot(
            params_specandphot, data_batch, polynomials_spec
        )

    @partial(jit, static_argnums=(2, 3))
    def update(step, opt_state, data_batch, data_aux):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_spec_and_specandphot)(
            params, data_batch, data_aux
        )
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
            losses_train[i, j], opt_state = update(
                next(itercount), opt_state, data_batch, polynomials_spec
            )
            # print("batch loss", losses_train[i, j])

            # batch_ells = train_specphotscaling[neworder[si : si + bs]]
            # update scaling, calculated from spec only
            # updated_batch_ells[updated_batch_ells < 0] = 1.0
            # train_specphotscaling[neworder[si : si + bs]] = updated_batch_ells

        loss = np.mean(losses_train[i, :])

        print(
            "> loss: %.5e" % loss,
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
            end=" - ",
        )

        pcamodel_speconly.write_model()
        pcamodel_specandphot.write_model()

        if i % output_valid_steps == 0:

            print("> Running validation models and data")
            resultspipe_speconly = ResultsPipeline(
                output_prefix,
                "_speconly",
                n_components + n_poly,
                datapipe,
                indices_valid,
            )
            resultspipe_specandphot = ResultsPipeline(
                output_prefix,
                "_specandphot",
                n_components + n_poly,
                datapipe,
                indices_valid,
            )

            datapipe.batch = 0  # reset batch number
            for j in range(numBatches_valid):
                data_batch = datapipe.next_batch(indices_valid, batchsize)

                result_speconly = pcamodel_speconly.bayesianpca_speconly(
                    params_speconly, data_batch, polynomials_spec
                )
                result_specandphot = pcamodel_specandphot.bayesianpca_specandphot(
                    params_specandphot, data_batch, polynomials_spec
                )

                resultspipe_speconly.write_batch(data_batch, *result_specandphot)
                resultspipe_specandphot.write_batch(data_batch, *result_specandphot)

                losses_valid[i, j] = -np.sum(result_speconly[0] + result_specandphot[0])

            current_validation_loss = np.mean(losses_valid[i, :])

            resultspipe_speconly.write_reconstructions()
            resultspipe_specandphot.write_reconstructions()
            del resultspipe_speconly
            del resultspipe_specandphot

            print("Validation loss: %.5e" % current_validation_loss)
            if (
                current_validation_loss > previous_validation_loss1
                and current_validation_loss > previous_validation_loss2
            ):
                print(
                    "Current validation loss if worse than the previous two - early stopping"
                )
                exit(0)
            previous_validation_loss2 = previous_validation_loss1
            previous_validation_loss1 = current_validation_loss

            np.save(output_dir + "/valid_losses", losses_valid[: i + 1, :])

    print("Learning finished")


if __name__ == "__main__":
    main()
