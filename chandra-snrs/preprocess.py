import logging
from pathlib import Path

import numpy as np
import requests
from astropy import units as u
from astropy.io import fits
from astropy.visualization import AsinhStretch
from astropy.wcs import WCS
from gammapy.estimators import ImageProfileEstimator
from gammapy.maps import Map, WcsGeom
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SIZE_REF = 1 * u.deg
BIN_SIZE_REF = 0.02 * u.deg
OVERSAMPLING_FACTOR = 2

URL_BASE = "https://hea-www.harvard.edu/ChandraSNR/{source}/{obs_id}/work/acis_E300-10000_FLUXED.fits.gz"


SOURCES = {
    "G292.0+01.8": 126,
    "G111.7-02.1": 114,
}


def download_file(url, filename):
    """Download file"""
    response = requests.get(url, stream=True)

    with Path(filename).open("wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


def read_map(filename):
    """Read map"""
    header = fits.getheader(filename)
    wcs = WCS(header)

    data = fits.getdata(filename)
    npix = (header["NAXIS1"], header["NAXIS2"])
    geom = WcsGeom(wcs=wcs, npix=npix)
    return Map.from_geom(geom=geom, data=data)


def download_data():
    """Download data"""

    for source, obs_id in SOURCES.items():
        filename = f"data/{source}-flux.fits.gz"

        if Path(filename).exists():
            log.info(f"Skipping {filename} as it already exists.")
            continue

        url = URL_BASE.format(source=source, obs_id=obs_id)
        download_file(url, filename=filename)


def measure_snr_size(data):
    """Measure SNR size from radial profile"""
    est = ImageProfileEstimator(axis="radial", center=data.geom.center_skydir)

    profile = est.run(data)
    smoothed = profile.smooth(kernel="gauss", radius=data.geom.pixel_scales[1])

    idx_max = np.argmax(smoothed.table["profile"])

    snr_size = profile.table["x_ref"].quantity[idx_max]
    return snr_size


def scale_image():
    """Scale image to a given SNR reference size"""
    for filename in Path("./data").glob("*.fits.gz"):
        data = read_map(filename=filename)
        size_snr = measure_snr_size(data)
        scaling_factor = (SIZE_REF / size_snr).to_value("")

        wcs = data.geom.wcs
        wcs.wcs.cdelt = wcs.wcs.cdelt * scaling_factor
        npix = data.geom.npix

        geom = WcsGeom(wcs=wcs, npix=npix)
        data = Map.from_geom(geom=geom, data=data.data)

        filename_out = Path(f"images/{filename.name}")
        filename_out.parent.mkdir(exist_ok=True, parents=True)
        log.info(f"Writing {filename_out}")

        geom = geom.to_binsz(binsz=BIN_SIZE_REF / OVERSAMPLING_FACTOR)

        data = data.smooth("0.01 deg").interp_to_geom(geom=geom)
        data.data = data.data / data.data.max()
        data.data = AsinhStretch(a=0.2)(data.data)
        data.write(filename_out, overwrite=True)


if __name__ == "__main__":
    download_data()
    scale_image()
