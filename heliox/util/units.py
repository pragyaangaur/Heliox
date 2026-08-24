"""
Units that solar physics uses but astropy does not define.

Importing this module registers the extra units with astropy's unit parser, so
that strings such as ``'sfu'`` taken from a FITS ``BUNIT`` keyword resolve
correctly.
"""

import astropy.units as u

__all__ = ["sfu", "dn", "register"]

#: The solar flux unit, used for radio measurements such as the F10.7 index.
sfu = u.def_unit(
    ["sfu", "solar flux unit"],
    1e-22 * u.W / (u.m**2 * u.Hz),
    doc="Solar flux unit: 10^-22 W / (m^2 Hz).",
)

#: Data number, the raw digitised output of a CCD. Astropy already defines this
#: one; it is re-exported here so that callers have a single place to look.
dn = u.DN

_registered = False


def register():
    """
    Register the heliox units so astropy's unit parser recognises them.

    Called automatically when this module is imported, and safe to call again.

    Examples
    --------
    >>> import astropy.units as u
    >>> import heliox.util.units  # doctest: +SKIP
    >>> u.Unit('sfu').to(u.W / (u.m**2 * u.Hz))
    1e-22
    """
    global _registered
    if not _registered:
        u.add_enabled_units([sfu])
        _registered = True


register()
