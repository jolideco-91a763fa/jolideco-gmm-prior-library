import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import invgamma

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PATH = Path(__file__).parent
RANDOM_STATE = np.random.RandomState(seed=9823)


DENSITY = 1 / 64  # sources per patch area
IMAGE_SIZE = 1024
N_IMAGES = 5
NOISE_FACTOR = 1e4
BKG_LEVEL = 1e-6


def grid_weights(x, y, x0, y0):
    """Compute 4-pixel weights such that centroid is preserved."""
    dx = np.abs(x - x0)
    dx = np.where(dx < 1, 1 - dx, 0)

    dy = np.abs(y - y0)
    dy = np.where(dy < 1, 1 - dy, 0)
    return dx * dy


def simulate_catalog(random_state=RANDOM_STATE):
    """Simulate a catalog of point sources"""
    n_sources = random_state.poisson(int(IMAGE_SIZE**2 * DENSITY))

    table = Table()
    table["x"] = random_state.uniform(0, IMAGE_SIZE - 1, n_sources)
    table["y"] = random_state.uniform(0, IMAGE_SIZE - 1, n_sources)
    table["flux"] = invgamma.rvs(1, scale=1, size=n_sources)
    return table


def catalog_to_image(sources, random_state=RANDOM_STATE, factor=1e-5):
    """Convert catalog to image"""
    image = factor * np.ones((IMAGE_SIZE, IMAGE_SIZE))

    x, y, flux = sources["x"], sources["y"], sources["flux"]
    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)

    image[y_int, x_int] = flux * grid_weights(x, y, x_int, y_int)
    image[y_int + 1, x_int] = flux * grid_weights(x, y, x_int, y_int + 1)
    image[y_int, x_int + 1] = flux * grid_weights(x, y, x_int + 1, y_int)
    image[y_int + 1, x_int + 1] = flux * grid_weights(x, y, x_int + 1, y_int + 1)

    image = random_state.gamma(NOISE_FACTOR * (image + BKG_LEVEL)) / NOISE_FACTOR
    return image / image.max()


def simulate_and_write_images():
    """Simulate and write images to disk"""

    for idx in range(N_IMAGES):
        sources = simulate_catalog()
        image = catalog_to_image(sources)

        hdu = fits.PrimaryHDU(image)
        filename = PATH / f"images/image_{idx}.fits.gz"
        filename.parent.mkdir(exist_ok=True)
        log.info(f"Writing {filename}")
        hdu.writeto(filename, overwrite=True)


if __name__ == "__main__":
    simulate_and_write_images()
