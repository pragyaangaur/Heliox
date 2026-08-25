"""Choosing a reader based on what a file actually is."""

from pathlib import Path

from heliox.io import _fits
from heliox.util.exceptions import UnrecognizedFileTypeError

__all__ = ["read_file", "read_file_header", "detect_filetype"]

#: The first few bytes that identify each supported format.
_MAGIC_NUMBERS = {
    b"SIMPLE  =": "fits",
    b"\x00\x00\x00\x0cjP  ": "jp2",
}

_EXTENSIONS = {
    ".fits": "fits",
    ".fit": "fits",
    ".fts": "fits",
    ".fits.gz": "fits",
}


def detect_filetype(filepath):
    """
    Work out a file's type from its contents, falling back to its extension.

    Reading the first few bytes is more reliable than trusting the name, since
    archives hand out FITS files with all sorts of extensions.

    Parameters
    ----------
    filepath : path-like
        The file to inspect.

    Returns
    -------
    `str`
        A short type name such as ``'fits'``.

    Raises
    ------
    `~heliox.util.exceptions.UnrecognizedFileTypeError`
        If the type could not be determined.

    Examples
    --------
    >>> from heliox.data.sample import AIA_171_IMAGE
    >>> from heliox.io import detect_filetype
    >>> detect_filetype(AIA_171_IMAGE)
    'fits'
    """
    path = Path(filepath)
    try:
        with open(path, "rb") as stream:
            head = stream.read(16)
    except OSError as exc:
        raise UnrecognizedFileTypeError(f"Could not open {filepath}: {exc}") from exc

    for magic, name in _MAGIC_NUMBERS.items():
        if head.startswith(magic):
            return name

    suffix = "".join(path.suffixes).lower()
    for extension, name in _EXTENSIONS.items():
        if suffix.endswith(extension):
            return name

    raise UnrecognizedFileTypeError(
        f"Could not work out the type of {filepath}. heliox reads FITS files; "
        "if this is one, it does not start with a valid FITS header."
    )


def read_file(filepath, filetype=None, **kwargs):
    """
    Read a file, choosing the reader automatically.

    Parameters
    ----------
    filepath : path-like
        The file to read.
    filetype : `str`, optional
        Force a particular reader instead of detecting one.
    **kwargs
        Passed through to the reader.

    Returns
    -------
    `list`
        One ``(data, header)`` pair per image in the file.
    """
    filetype = filetype or detect_filetype(filepath)
    if filetype == "fits":
        return _fits.read(filepath, **kwargs)
    raise UnrecognizedFileTypeError(f"heliox has no reader for {filetype!r} files.")


def read_file_header(filepath, filetype=None, **kwargs):
    """
    Read only the headers of a file, without loading any image data.

    Parameters
    ----------
    filepath : path-like
        The file to read.
    filetype : `str`, optional
        Force a particular reader.
    **kwargs
        Passed through to the reader.

    Returns
    -------
    `list` of `~heliox.util.metadata.MetaDict`
    """
    from astropy.io import fits

    filetype = filetype or detect_filetype(filepath)
    if filetype != "fits":
        raise UnrecognizedFileTypeError(f"heliox has no reader for {filetype!r} files.")

    with fits.open(filepath, **kwargs) as hdulist:
        return [_fits.get_header(hdu) for hdu in hdulist]
