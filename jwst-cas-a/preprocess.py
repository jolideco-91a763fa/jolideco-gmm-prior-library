import logging

from astropy.io import fits
from matplotlib.image import imread
from skimage.color import rgb2gray

log = logging.getLogger(__name__)


def read_image(filename="data/jwst-cas-a.png"):
    """Read image and convert to gray"""
    log.info(f"Reading {filename}")
    image = imread(filename)
    image = rgb2gray(image)
    return image


if __name__ == "__main__":
    image = read_image()
    fits.writeto("images/jwst-cas-a.fits.gz", image, overwrite=True)
