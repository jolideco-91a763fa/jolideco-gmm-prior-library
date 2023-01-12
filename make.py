import json
import logging
import subprocess
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits
from astropy.nddata import block_reduce
from jolideco.priors.patches import GaussianMixtureModel
from jolideco.priors.patches.train import sklearn_gmm_to_table
from jolideco.utils.numpy import view_as_overlapping_patches
from scipy.ndimage import gaussian_filter
from skimage.transform import rotate
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RANDOM_STATE = np.random.RandomState(seed=0)


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        log.info(f"Execution time: {t1 - t0:.2f} s")
        return result

    return wrapper


@click.group()
def cli():
    pass


def read_config(filename):
    """Read config file"""
    log.info(f"Reading {filename}")
    data = yaml.safe_load(filename.read_text())
    return data


@cli.command("pre-process-images")
@click.argument("filename", type=Path)
@timeit
def pre_process_images(filename):
    """Pre-process data using a custom script"""
    config = read_config(filename)

    command = ["python", config["pre-process-script"]]
    log.info(f"Running command: {' '.join(command)}")
    subprocess.call(command, cwd=filename.parent)


def read_images(config, path_base):
    """Read images from FITS files"""
    images = []

    for filename in path_base.glob(config["extract-patches"]["input-images"]):
        log.info(f"Reading {filename}")
        image = fits.getdata(filename)
        images.append(image)

    return images


def apply_random_rotation(image):
    """Apply a random rotation to images

    Parameters
    ----------
    images : list of `~numpy.ndarray`
        List of images

    Returns
    -------
    images_rotated : list of `~numpy.ndarray`
        List of rotated images
    """
    theta = RANDOM_STATE.uniform(0, 180)  # in deg
    image_rotated = rotate(image, angle=theta, resize=True)
    return image_rotated


@cli.command("extract-patches")
@click.argument("filename", type=Path)
@timeit
def extract_patches(filename):
    """Read images and extract and save patches"""
    config = read_config(filename=filename)
    images = read_images(config=config, path_base=filename.parent)

    patches = []

    stride = config["extract-patches"]["stride"]
    patch_shape = tuple(config["extract-patches"]["patch-shape"])
    sigma = config["extract-patches"]["gaussian-smoothing"]
    block_size = config["extract-patches"]["downsample-block-size"]

    for image in images:
        smoothed = gaussian_filter(image, sigma)
        image_coarse = block_reduce(smoothed, block_size, func=np.mean)

        if config["extract-patches"].get("random-rotation", False):
            image_coarse = apply_random_rotation(image_coarse)

        p = view_as_overlapping_patches(image_coarse, shape=patch_shape, stride=stride)
        valid = np.all(p > 0, axis=1)
        patches.extend(p[valid])

    patches = np.array(patches)

    patches_normed = patches - patches.mean(
        axis=1, keepdims=True
    )  # / patches.std(axis=1, keepdims=True)

    filename_patches = filename.parent / config["extract-patches"]["filename"]
    log.info(f"Writing {filename_patches}")
    fits.writeto(filename_patches, data=patches_normed, overwrite=True)


@cli.command("learn-gmm")
@click.argument("filename", type=Path)
@timeit
def learn_gmm(filename):
    """Learn a Gaussian Mixture Model from a list of patches"""
    config = read_config(filename=filename)

    gmm = GaussianMixture(**config["sklearn-gmm-kwargs"])

    filename_patches = filename.parent / config["extract-patches"]["filename"]
    log.info(f"Reading {filename_patches}")
    patches = fits.getdata(filename_patches)

    log.info(f"Fitting GMM to {len(patches)} patches...")
    gmm.fit(X=patches)

    filename_gmm = filename.parent / config["filename"]
    table = sklearn_gmm_to_table(gmm=gmm)
    log.info(f"Writing {filename_gmm}")
    table.write(filename_gmm, overwrite=True)


def plot_example_patches(patches, patch_shape, n_patches=60):
    """Plot example patches"""
    _, axes = plt.subplots(ncols=10, nrows=6, figsize=(10, 8))

    idx_list = RANDOM_STATE.randint(0, len(patches), n_patches)

    for idx, ax in zip(idx_list, axes.flat):
        ax.imshow(patches[idx].reshape(patch_shape))
        ax.set_axis_off()
        ax.set_title(f"{idx}")

    filename = "patches-examples.png"
    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=300)


@cli.command("summarize-gmm")
@click.argument("filename", type=Path)
def summarize_gmm(filename):
    """Summarize a Gaussian Mixture Model"""
    config = read_config(filename)

    gmm = GaussianMixtureModel.read(
        filename=filename.parent / config["filename"], format=config["format"]
    )

    gmm.plot_mean_images(ncols=config["plots"]["ncols"])
    plt.tight_layout()

    path = filename.parent / "plots"
    path.mkdir(exist_ok=True)

    name = config["name"]
    filename = path / f"gmm-means-{name}.png"
    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=300)

    gmm.plot_eigen_images(ncols=config["plots"]["ncols"])
    plt.tight_layout()

    filename = path / f"gmm-eigen-images-{name}.png"
    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=300)


@cli.command("write-index")
def write_index_file():
    """Write index file

    Read all config files from sub-directories and write a json index file with all entries.
    """
    index = {}

    for filename in Path(".").glob("*/config*.yaml"):
        log.info(f"Reading {filename}")
        data = yaml.safe_load(filename.read_text())

        filename_gmm = "$JOLIDECO_GMM_LIBRARY/" + str(
            filename.parent / data["filename"]
        )
        entry = {"filename": filename_gmm, "format": data["format"]}
        index[data["name"]] = entry

    filename = Path("jolideco-gmm-library-index.json")

    with filename.open("w") as fh:
        log.info(f"Writing {filename}")
        json.dump(index, fh, indent=2)


if __name__ == "__main__":
    cli()
