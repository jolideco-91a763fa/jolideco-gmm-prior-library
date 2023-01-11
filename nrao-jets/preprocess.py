import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.visualization import AsinhStretch
from jolideco.utils.numpy import view_as_overlapping_patches
from matplotlib.image import imread
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from skimage.color import rgb2gray
from skimage.feature import blob_dog
from skimage.transform import AffineTransform, resize, rotate, warp
from skimage.util import view_as_blocks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_jet_images(filename="data/radio-jets-samples.jpg", size=(54, 64)):
    """Reading the jets, jets jets image and split into smaller images

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    images : `~numpy.ndarray`
        Images
    """
    log.info(f"Reading {filename}")
    image = imread(filename)
    image = rgb2gray(image)

    blocks = view_as_blocks(image, block_shape=size)
    blocks = blocks.reshape((-1,) + size)
    images = blocks[:, slice(2, 50), slice(2, 63)]
    return images


def shift_image(image, vector):
    """Shift image

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image
    vector : tuple
        Shift vector

    Returns
    -------
    shifted : `~numpy.ndarray`
        Shifted image
    """
    transform = AffineTransform(translation=vector)
    shifted = warp(
        image,
        transform,
        mode="wrap",
        preserve_range=True,
    )
    return shifted


def align_main_axis(image, output_shape=(78, 78), threshold=0.05):
    """Align main axis

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image
    output_shape : tuple
        Output shape
    threshold : float
        Threshold for blob detection

    Returns
    -------
    image : `~numpy.ndarray`
        Image
    """
    blobs = blob_dog(image=image, max_sigma=5, threshold=threshold)

    if len(blobs) < 2:
        return None

    y_blob, x_blob = blobs.T[0], blobs.T[1]

    vector = np.mean(y_blob) - image.shape[0] / 2, np.mean(x_blob) - image.shape[1] / 2
    image = shift_image(image, vector[::-1])

    result = linregress(x_blob, y_blob)
    theta = np.rad2deg(np.arctan2(result.slope, 1))
    image = rotate(image, angle=theta, resize=True)

    ny, nx = image.shape
    x = np.arange(nx) - (nx - 1) / 2  # np.arange(image.shape[1]) - np.mean(y_blob)
    y = np.arange(ny) - (ny - 1) / 2  # np.arange(image.shape[0]) - np.mean(x_blob)
    interp = interp2d(x=x, y=y, z=image)

    ny, nx = output_shape
    x_new = np.arange(nx) - (nx - 1) / 2
    y_new = np.arange(ny) - (ny - 1) / 2
    image = resize(interp(y=y_new, x=x_new), output_shape=(128, 128))
    return image[slice(32, 92), :]


def renormalize_image(image, stretch=AsinhStretch(a=0.2), threshold=0.12):
    """Renormalize image"""
    normed = stretch(image)
    normed[normed < threshold] = 0
    return normed


def align_images(images, output_shape=(88, 88)):
    """Align images"""
    images_aligned = []

    for image in images:
        aligned = align_main_axis(image, output_shape=output_shape)

        if aligned is not None:
            normed = renormalize_image(aligned)
            images_aligned.append(normed)

    return images_aligned


def save_images(images):
    """Save images"""
    path = Path("images")
    path.mkdir(exist_ok=True)

    for idx, image in enumerate(images):
        filename = path / f"image-nrao-jet-{idx:03d}.fits.gz"
        log.info(f"Writing {filename}")
        fits.writeto(filename, data=image, overwrite=True)


def pre_process_images():
    """Pre-process images"""
    images = get_jet_images()
    images_aligned = align_images(images)
    save_images(images_aligned)


def apply_random_rotation(images):
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
    images_rotated = []

    for image in images:
        theta = np.random.uniform(-10, 10)
        image_rotated = rotate(image, angle=theta, resize=True)
        images_rotated.append(image_rotated)

    return images_rotated


if __name__ == "__main__":
    pre_process_images()
