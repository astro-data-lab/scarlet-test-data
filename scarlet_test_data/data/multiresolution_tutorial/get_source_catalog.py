
import os
from jax import vmap
import jax.numpy as jnp
import numpy as np

import astropy.io.fits as fits
from astropy.wcs import WCS

import scarlet2
from scarlet2 import Starlet

# Installing data package if not already installed
from scarlet2.utils import import_scarlet_test_data

import_scarlet_test_data()
from scarlet_test_data import data_path

import sep

from astropy.table import Table

# Load the HSC image data
obs_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HSC1.fits"))
data_hsc = obs_hdu[0].data.astype('float32')
wcs_hsc = WCS(obs_hdu[0].header)
channels_hsc = ['g', 'r', 'i', 'z', 'y']

# Load the HSC PSF data
psf_hsc_data = fits.open(os.path.join(data_path, "test_resampling", "PSF_HSC.fits"))[0].data.astype('float32')
psf_hsc = scarlet2.ArrayPSF(psf_hsc_data)

# Load the HST image data
hst_hdu = fits.open(os.path.join(data_path, "test_resampling", "Cut_HST1.fits"))
data_hst = hst_hdu[0].data.astype('float32')
wcs_hst = WCS(hst_hdu[0].header)
channels_hst = ['F814W']

# Load the HST PSF data
psf_hst_data = fits.open(os.path.join(data_path, "test_resampling", "PSF_HST.fits"))[0].data.astype('float32')
psf_hst_data = psf_hst_data[None, :, :]
psf_hst = scarlet2.ArrayPSF(psf_hst_data)

# Scale the HST data
data_hst = data_hst[None, ...].astype('float32')
data_hst *= data_hsc.max() / data_hst.max()

"""
Next we have to create a source catalog for the images. We’ll use `sep` for that, 
but any other detection method will do. Since HST is higher resolution and less affected by blending, 
we use it for detection but we also run detection on the HSC image to calculate the background RMS
"""

def makeCatalog(data_lr, data_hr, lvl=3, wave=True):
    # Create a catalog of detected source by running SEP on the wavelet transform
    # of the sum of the high resolution images and the low resolution images interpolated 
    # to the high resolution grid

    coords_in = jnp.stack(jnp.meshgrid(jnp.linspace(0, 1, data_lr.shape[-1] + 2)[1:-1],
                                       jnp.linspace(0, 1, data_lr.shape[-2] + 2)[1:-1]), -1)

    coords_out = jnp.stack(jnp.meshgrid(jnp.linspace(0, 1, data_hr.shape[-2] + 2)[1:-1],
                                      jnp.linspace(0, 1, data_hr.shape[-2] + 2)[1:-1]), -1)

    interp_im = vmap(scarlet2.interpolation.resample2d,
                         in_axes=(0, None, None))(data_lr, coords_in, coords_out)

    # Normalisation
    interp_im = interp_im / jnp.sum(interp_im, axis=(1, 2))[:, None, None]
    hr_images = data_hr / jnp.sum(data_hr, axis=(1, 2))[:, None, None]

    # Summation to create a detection image
    detect_image = jnp.sum(interp_im, axis=0) + jnp.sum(hr_images, axis=0)
    # Rescaling to HR image flux
    detect_image *= jnp.sum(data_hr)
    # Wavelet transform
    wave_detect = Starlet.from_image(detect_image).coefficients

    if wave:
        # Creates detection from the first 3 wavelet levels
        detect = wave_detect[:lvl, :, :].sum(axis=0)
    else:
        detect = detect_image

    # Runs SEP detection
    bkg = sep.Background(np.array(detect))
    catalog = sep.extract(np.array(detect), 3, err=bkg.globalrms)
    bg_rms = []
    for img in [np.array(data_lr), np.array(data_hr)]:
        if np.size(img.shape) == 3:
            bg_rms.append(np.array([sep.Background(band).globalrms for band in img]))
        else:
            bg_rms.append(sep.Background(img).globalrms)

    return catalog, bg_rms, detect_image

# Making catalog.
# With the wavelet option on, only the first 3 wavelet levels are used for detection. Set to 1 for better detection
wave = 1
lvl = 3
catalog_hst, (bg_hsc, bg_hst), detect = makeCatalog(data_hsc, data_hst, lvl, wave)


# we can now set the empirical noise rms for both observations
obs_hst_weights = np.ones(data_hst.shape) / (bg_hst ** 2)[:, None, None]
obs_hsc_weights = np.ones(data_hsc.shape) / (bg_hsc ** 2)[:, None, None]

obs_hst = scarlet2.Observation(data_hst,
                               wcs=wcs_hst,
                               psf=psf_hst,
                               channels=channels_hst,
                               weights=obs_hst_weights)

pixel_hst = np.stack((catalog_hst['y'], catalog_hst['x']), axis=1)
# Convert the HST pixel coordinates to sky coordinates
coords = obs_hst.frame.get_sky_coord(pixel_hst)

# Saving everything in a single fits file

# Coordinates table (FK5 J2000.0)
table = Table([coords.ra.deg, coords.dec.deg], names=('RA', 'DEC'))
coord_hdu = fits.BinTableHDU(data=table)
coord_hdu.header['RADECSYS'] = coords.frame.name.upper() # 'FK5'
coord_hdu.header['EQUINOX'] = coords.equinox.jyear # 2000.0

# Build HDUs images
primary_hdu = fits.PrimaryHDU(data=data_hsc, header=wcs_hsc.to_header())  # Observation 1 Image
primary_hdu.header['EXTNAME'] = 'HSC_OBS'

psf1_hdu = fits.ImageHDU(data=psf_hsc_data)
psf1_hdu.header['EXTNAME'] = 'HSC_PSF'

weight1_hdu = fits.ImageHDU(data=obs_hsc_weights)
weight1_hdu.header['EXTNAME'] = 'HSC_WEIGHTS'

image2_hdu = fits.ImageHDU(data=data_hst, header=wcs_hst.to_header())
image2_hdu.header['EXTNAME'] = 'HST_OBS'

psf2_hdu = fits.ImageHDU(data=psf_hst_data)
psf2_hdu.header['EXTNAME'] = 'HST_PSF'

weight2_hdu = fits.ImageHDU(data=obs_hst_weights)
weight2_hdu.header['EXTNAME'] = 'HST_WEIGHTS'

coord_hdu.header['EXTNAME'] = 'CATALOG'

# Bundle all HDUs
hdulist = fits.HDUList([
    primary_hdu,
    psf1_hdu,
    weight1_hdu,
    image2_hdu,
    psf2_hdu,
    weight2_hdu,
    coord_hdu
])

hdulist.writeto('multiresolution_tutorial_data.fits', overwrite=True)