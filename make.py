import json
import logging
import subprocess
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jolideco.priors.patches.train import sklearn_gmm_to_table
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RANDOM_STATE = np.random.RandomState(seed=0)


@click.group()
def cli():
    pass


def read_config(filename):
    """Read config file"""
    log.info(f"Reading {filename}")
    data = yaml.safe_load(filename.read_text())
    return data


@cli.command("pre-process")
@cli.argument("filename", type=click.Path(exists=True))
def pre_process(filename):
    """Pre-process data using a custom script"""
    config = read_config(filename)

    command = ["python", config["pre-process-script"]]
    log.info(f"Running {command}")
    subprocess.call(command)


def prepare_patches(config):
    """"""
    pass


@cli.command("learn-gmm")
@cli.argument("filename", type=click.Path(exists=True))
def learn_gmm_model(filename):
    """Learn a Gaussian Mixture Model from a list of patches"""
    config = yaml.safe_load(filename)

    gmm = GaussianMixture(**config["learn-gmm"]["sklearn-gmm-kwargs"])

    patches = prepare_patches(config["learn-gmm"]["patches"])

    patches_normed = patches - patches.mean(
        axis=1, keepdims=True
    )  # / patches.std(axis=1, keepdims=True)

    gmm.fit(X=patches_normed)

    filename = config["filename"]
    table = sklearn_gmm_to_table(gmm=gmm)
    log.info(f"Writing {filename}")
    table.write(filename, overwrite=True)


def plot_example_patches(patches, patch_shape, n_patches=60):
    """Plot example patches"""
    fig, axes = plt.subplots(ncols=10, nrows=6, figsize=(10, 8))

    idx_list = RANDOM_STATE.randint(0, len(patches), n_patches)

    for idx, ax in zip(idx_list, axes.flat):
        ax.imshow(patches[idx].reshape(patch_shape))
        ax.set_axis_off()
        ax.set_title(f"{idx}")

    filename = "patches-examples.png"
    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=300)


def plot_gmm_means(gmm):
    """"""
    pass



@cli.command("summarize-gmm")
@cli.argument("filename", type=click.Path(exists=True))
def summarize_gmm(filename):
    """Summarize a Gaussian Mixture Model"""
    config = read_config(filename)

    plot_example_patches(patches=, patch_shape=, n_patches=)


@cli.command("write-index")
def write_index_file():
    """Write index file

    Read all config files from sub-directories and write a json index file with all entries.
    """
    index = {}

    for filename in Path(".").glob("*/config.yaml"):
        log.info(f"Reading {filename}")
        data = yaml.safe_load(filename.read_text())

        filename_gmm = "$JOLIDECO_GMM_LIBRARAY" + filename.parent / data["filename"]
        entry = {"filename": filename_gmm, "format": data["format"]}
        index[data["name"]] = entry

    filename = Path("index.json")

    with filename.open("w") as fh:
        log.info(f"Writing {filename}")
        json.dump(index, fh, indent=2)


if __name__ == "__main__":
    cli()
