import logging
from pathlib import Path

from astropy import units as u
from gammapy.maps import HpxGeom, WcsGeom
from gleam_client import vo_get

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ANG_SIZE = 5
HPX_GEOM = HpxGeom.create(binsz=3 * u.deg)
GC_PLANE = WcsGeom.create(skydir=(0, 0), frame="galactic", width=(180, 12))

PROJECTION = "ZEA"

FREQUENCIES = [
    "170-231",
]


def create_filename(ra, dec, ang_size, freq, error=False):
    """
    You can write your own create_filename function however you like
    Here is a dummy example
    """
    if error:
        return "error_{0:.0f}_{1:.0f}.html".format(ra, dec)
    else:
        return f"{freq}MHz-ra-{ra:.1f}-dec-{dec:.1f}-stamp.fits"


def download_stamps():
    coords = HPX_GEOM.get_coord().skycoord
    selection = GC_PLANE.contains(coords)

    for skydir in coords[selection]:
        filename = create_filename(
            ra=skydir.icrs.ra.deg,
            dec=skydir.icrs.dec.deg,
            ang_size=ANG_SIZE,
            freq=FREQUENCIES[0],
        )

        path = Path("images") / filename

        if path.exists() or path.with_suffix(".fits.gz").exists():
            log.info(f"Skipping {path} as it already exists.")
            continue

        vo_get(
            ra=skydir.icrs.ra.deg,
            dec=skydir.icrs.dec.deg,
            ang_size=ANG_SIZE,
            download_dir="images",
            projection=PROJECTION,
            file_name_func=create_filename,
            freq=FREQUENCIES,
            clobber=True,
        )


if __name__ == "__main__":
    download_stamps()
