"""
Reading and writing FITS files.

A thin layer over `astropy.io.fits` that returns data and metadata in the form
`heliox.map` and `heliox.timeseries` expect, and that copes with the
imperfections common in solar FITS files: non-standard keywords, blank cards,
and headers that fail verification.
"""

import collections
import warnings

import numpy as np

from astropy.io import fits

from heliox.util.exceptions import HelioxMetadataWarning
from heliox.util.metadata import MetaDict

__all__ = ["read", "write", "get_header", "header_to_fits", "extract_waveunit"]

#: A single image and its metadata, as returned by `read`.
HDUPair = collections.namedtuple("HDUPair", ["data", "header"])


def read(filepath, hdus=None, memmap=None, **kwargs):
    """
    Read every image extension of a FITS file.

    Parameters
    ----------
    filepath : path-like
        The file to read.
    hdus : `int` or iterable of `int`, optional
        Which extensions to read. By default every extension that contains
        data is returned.
    memmap : `bool`, optional
        Passed to `astropy.io.fits.open`.
    **kwargs
        Passed to `astropy.io.fits.open`.

    Returns
    -------
    `list` of `HDUPair`
        One ``(data, header)`` pair per extension, in file order.

    Examples
    --------
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.io._fits import read
    >>> pairs = read(AIA_171_IMAGE)
    >>> pairs[0].data.shape
    (512, 512)
    >>> pairs[0].header['instrume']
    'AIA'
    """
    with fits.open(filepath, memmap=memmap, ignore_blank=True, **kwargs) as hdulist:
        hdulist.verify("silentfix+warn")

        if hdus is None:
            selected = list(hdulist)
        elif isinstance(hdus, int):
            selected = [hdulist[hdus]]
        else:
            selected = [hdulist[index] for index in hdus]

        pairs = []
        for hdu in selected:
            if hdu.data is None:
                continue
            pairs.append(HDUPair(hdu.data, get_header(hdu)))
        return pairs


def get_header(hdu):
    """
    Convert a FITS header into a `~heliox.util.metadata.MetaDict`.

    Comment and history cards are collapsed into single ``'comment'`` and
    ``'history'`` entries, and blank cards are dropped.

    Parameters
    ----------
    hdu : `astropy.io.fits.hdu.base.ExtensionHDU` or `astropy.io.fits.Header`
        The HDU or header to convert.

    Returns
    -------
    `~heliox.util.metadata.MetaDict`
    """
    header = hdu.header if hasattr(hdu, "header") else hdu

    meta = MetaDict()
    comments = []
    history = []

    for card in header.cards:
        key = card.keyword
        if key == "COMMENT":
            comments.append(str(card.value).strip())
        elif key == "HISTORY":
            history.append(str(card.value).strip())
        elif key == "":
            continue
        else:
            meta[key] = card.value

    if comments:
        meta["comment"] = "\n".join(comments)
    if history:
        meta["history"] = "\n".join(history)

    waveunit = extract_waveunit(meta)
    if waveunit is not None:
        meta["waveunit"] = waveunit

    return meta


def header_to_fits(header):
    """
    Convert a metadata mapping back into a `astropy.io.fits.Header`.

    Keys that FITS cannot represent -- names longer than eight characters
    without a ``HIERARCH`` prefix, or values of unsupported types -- are
    dropped with a warning rather than causing the write to fail.

    Parameters
    ----------
    header : mapping
        The metadata to convert.

    Returns
    -------
    `astropy.io.fits.Header`
    """
    fits_header = fits.Header()
    for key, value in header.items():
        if key.lower() in ("comment", "history"):
            for line in str(value).split("\n"):
                fits_header.add_comment(line) if key.lower() == "comment" else (
                    fits_header.add_history(line)
                )
            continue
        if isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        try:
            fits_header.append(fits.Card(key.upper(), value))
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"The keyword {key!r} could not be written to FITS and was dropped: {exc}",
                HelioxMetadataWarning,
                stacklevel=3,
            )
    return fits_header


def write(filepath, data, header, hdu_type=None, **kwargs):
    """
    Write an array and its metadata to a FITS file.

    Parameters
    ----------
    filepath : path-like
        Where to write the file.
    data : `numpy.ndarray`
        The image to write.
    header : mapping
        The metadata to write, converted with `header_to_fits`.
    hdu_type : `astropy.io.fits.hdu.base.ExtensionHDU`, optional
        The HDU class to use. Defaults to `astropy.io.fits.PrimaryHDU`.
    **kwargs
        Passed to `astropy.io.fits.HDUList.writeto`; ``overwrite=True`` is a
        common one.
    """
    fits_header = header_to_fits(header)
    # These describe the array itself and are rewritten by astropy, so a stale
    # copy carried over from the input header would be wrong.
    for key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND"):
        fits_header.pop(key, None)

    hdu_type = fits.PrimaryHDU if hdu_type is None else hdu_type
    hdu = hdu_type(data=data, header=fits_header)
    fits.HDUList([hdu]).writeto(filepath, **kwargs)


#: Recognised spellings of wavelength units, mapped onto their canonical names.
_WAVEUNITS = {
    "angstrom": "Angstrom",
    "angstroms": "Angstrom",
    "a": "Angstrom",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "um": "um",
    "micron": "um",
    "micrometer": "um",
    "mm": "mm",
    "m": "m",
}


def extract_waveunit(meta):
    """
    Work out the unit of the ``WAVELNTH`` keyword.

    Instruments record this inconsistently: sometimes in a ``WAVEUNIT``
    keyword, sometimes only in the free-text comment attached to ``WAVELNTH``.
    Both are checked.

    Parameters
    ----------
    meta : mapping
        The metadata to inspect.

    Returns
    -------
    `str` or `None`
        The canonical unit name, or `None` if it could not be determined.

    Examples
    --------
    >>> from heliox.io._fits import extract_waveunit
    >>> extract_waveunit({'waveunit': 'angstroms'})
    'Angstrom'
    >>> extract_waveunit({'wavelnth': 171}) is None
    True
    """
    raw = meta.get("waveunit")
    if raw is not None:
        candidate = str(raw).strip().lower()
        return _WAVEUNITS.get(candidate, str(raw).strip())

    comment = meta.get("comment")
    if comment:
        for word in str(comment).lower().replace(",", " ").split():
            if word in _WAVEUNITS:
                return _WAVEUNITS[word]
    return None
