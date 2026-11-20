// Copyright and licence notice of ERFA, reproduced from the ERFA source files that this module follows (src/nut00a.c, src/nut06a.c, src/obl06.c and src/fal03.c and the other fundamental argument files):
//
//  Copyright (C) 2013-2023, NumFOCUS Foundation.
//  All rights reserved.
//
//  This library is derived, with permission, from the International
//  Astronomical Union's "Standards of Fundamental Astronomy" library,
//  available from http://www.iausofa.org.
//
//  The ERFA version is intended to retain identical functionality to
//  the SOFA library, but made distinct through different function and
//  file names, as set out in the SOFA license conditions.  The SOFA
//  original has a role as a reference standard for the IAU and IERS,
//  and consequently redistribution is permitted only in its unaltered
//  state.  The ERFA version is not subject to this restriction and
//  therefore can be included in distributions which do not support the
//  concept of "read only" software.
//
//  Although the intent is to replicate the SOFA API (other than
//  replacement of prefix names) and results (with the exception of
//  bugs;  any that are discovered will be fixed), SOFA is not
//  responsible for any errors found in this version of the library.
//
//  If you wish to acknowledge the SOFA heritage, please acknowledge
//  that you are using a library derived from SOFA, rather than SOFA
//  itself.
//
//
//  TERMS AND CONDITIONS
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  1 Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  2 Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//
//  3 Neither the name of the Standards Of Fundamental Astronomy Board,
//    the International Astronomical Union nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
//  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

//! Nutation and the mean obliquity of the ecliptic, from ERFA.
//!
//! This file reproduces ERFA's `eraNut06a`, `eraNut00a`, `eraObl06` and the
//! fundamental argument routines that `eraNut00a` calls (`eraFal03`, `eraFaf03`,
//! `eraFaom03`, `eraFapa03`, `eraFame03`, `eraFave03`, `eraFae03`, `eraFama03`,
//! `eraFaju03`, `eraFasa03` and `eraFaur03`). Each function's doc comment names the
//! ERFA routine it follows. The operations keep the order of the C code, so that the
//! rounding matches. The coefficient tables are in `nutation/coefficients.rs`.

mod coefficients;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::{check_same_len, Array};
use coefficients::{LUNI_SOLAR, PLANETARY};

/// The Julian date of J2000.0, ERFA's `ERFA_DJ00`.
const DJ00: f64 = 2451545.0;
/// The number of days in a Julian century, ERFA's `ERFA_DJC`.
const DJC: f64 = 36525.0;
/// Two pi, ERFA's `ERFA_D2PI`.
const D2PI: f64 = std::f64::consts::TAU;
/// Arcseconds in a full turn, ERFA's `ERFA_TURNAS`.
const TURNAS: f64 = 1296000.0;
/// Radians per arcsecond, ERFA's `ERFA_DAS2R`.
const DAS2R: f64 = 4.84813681109536e-6;
/// Radians per 0.1 microarcsecond, the unit of the coefficient tables.
const U2R: f64 = DAS2R / 1e7;

/// Julian centuries of TT since J2000.0 for the date `jd1 + jd2`.
fn centuries_since_j2000(jd1: f64, jd2: f64) -> f64 {
    ((jd1 - DJ00) + jd2) / DJC
}

/// Reduces an angle in arcseconds modulo a full turn and converts it to radians.
fn turnas_to_radians(arcsec: f64) -> f64 {
    (arcsec % TURNAS) * DAS2R
}

/// Mean anomaly of the Moon, following `eraFal03` (IERS 2003).
fn moon_anomaly(t: f64) -> f64 {
    turnas_to_radians(
        485868.249036 + t * (1717915923.2178 + t * (31.8792 + t * (0.051635 + t * (-0.00024470)))),
    )
}

/// Mean longitude of the Moon minus that of its ascending node, following `eraFaf03`.
fn moon_argument_of_latitude(t: f64) -> f64 {
    turnas_to_radians(
        335779.526232 + t * (1739527262.8478 + t * (-12.7512 + t * (-0.001037 + t * (0.00000417)))),
    )
}

/// Mean longitude of the Moon's ascending node, following `eraFaom03`.
fn moon_ascending_node(t: f64) -> f64 {
    turnas_to_radians(
        450160.398036 + t * (-6962890.5431 + t * (7.4722 + t * (0.007702 + t * (-0.00005939)))),
    )
}

/// Mean elongation of the Moon from the Sun, following the MHB2000 polynomial in `eraNut00a`.
fn moon_elongation(t: f64) -> f64 {
    turnas_to_radians(
        1072260.70369 + t * (1602961601.2090 + t * (-6.3706 + t * (0.006593 + t * (-0.00003169)))),
    )
}

/// Mean anomaly of the Sun, following the MHB2000 polynomial in `eraNut00a`.
fn sun_anomaly(t: f64) -> f64 {
    turnas_to_radians(
        1287104.79305 + t * (129596581.0481 + t * (-0.5532 + t * (0.000136 + t * (-0.00001149)))),
    )
}

/// General accumulated precession in longitude, following `eraFapa03`.
fn general_precession(t: f64) -> f64 {
    (0.024381750 + 0.00000538691 * t) * t
}

/// Mean longitude of Mercury, following `eraFame03`.
fn mercury_longitude(t: f64) -> f64 {
    (4.402608842 + 2608.7903141574 * t) % D2PI
}

/// Mean longitude of Venus, following `eraFave03`.
fn venus_longitude(t: f64) -> f64 {
    (3.176146697 + 1021.3285546211 * t) % D2PI
}

/// Mean longitude of the Earth, following `eraFae03`.
fn earth_longitude(t: f64) -> f64 {
    (1.753470314 + 628.3075849991 * t) % D2PI
}

/// Mean longitude of Mars, following `eraFama03`.
fn mars_longitude(t: f64) -> f64 {
    (6.203480913 + 334.0612426700 * t) % D2PI
}

/// Mean longitude of Jupiter, following `eraFaju03`.
fn jupiter_longitude(t: f64) -> f64 {
    (0.599546497 + 52.9690962641 * t) % D2PI
}

/// Mean longitude of Saturn, following `eraFasa03`.
fn saturn_longitude(t: f64) -> f64 {
    (0.874016757 + 21.3299104960 * t) % D2PI
}

/// Mean longitude of Uranus, following `eraFaur03`.
fn uranus_longitude(t: f64) -> f64 {
    (5.481293872 + 7.4781598567 * t) % D2PI
}

/// The luni-solar and planetary nutation in longitude and obliquity, following `eraNut00a`.
///
/// Returns `(dpsi, deps)` in radians, without the P03 adjustments of `eraNut06a`.
fn nut00a(t: f64) -> (f64, f64) {
    let el = moon_anomaly(t);
    let elp = sun_anomaly(t);
    let f = moon_argument_of_latitude(t);
    let d = moon_elongation(t);
    let om = moon_ascending_node(t);

    let (mut dp, mut de) = (0.0, 0.0);
    for row in LUNI_SOLAR.iter().rev() {
        let [nl, nlp, nf, nd, nom, sp, spt, cp, ce, cet, se] = *row;
        let arg = (nl * el + nlp * elp + nf * f + nd * d + nom * om) % D2PI;
        let (sarg, carg) = (arg.sin(), arg.cos());
        dp += (sp + spt * t) * sarg + cp * carg;
        de += (ce + cet * t) * carg + se * sarg;
    }
    let (dpsils, depsls) = (dp * U2R, de * U2R);

    let al = (2.35555598 + 8328.6914269554 * t) % D2PI;
    let af = (1.627905234 + 8433.466158131 * t) % D2PI;
    let ad = (5.198466741 + 7771.3771468121 * t) % D2PI;
    let aom = (2.18243920 - 33.757045 * t) % D2PI;
    let apa = general_precession(t);
    let alme = mercury_longitude(t);
    let alve = venus_longitude(t);
    let alea = earth_longitude(t);
    let alma = mars_longitude(t);
    let alju = jupiter_longitude(t);
    let alsa = saturn_longitude(t);
    let alur = uranus_longitude(t);
    let alne = (5.321159000 + 3.8127774000 * t) % D2PI;

    let (mut dp, mut de) = (0.0, 0.0);
    for row in PLANETARY.iter().rev() {
        let [nl, nf, nd, nom, nme, nve, nea, nma, nju, nsa, nur, nne, npa, sp, cp, se, ce] = *row;
        let arg = (nl * al
            + nf * af
            + nd * ad
            + nom * aom
            + nme * alme
            + nve * alve
            + nea * alea
            + nma * alma
            + nju * alju
            + nsa * alsa
            + nur * alur
            + nne * alne
            + npa * apa)
            % D2PI;
        let (sarg, carg) = (arg.sin(), arg.cos());
        dp += sp * sarg + cp * carg;
        de += se * sarg + ce * carg;
    }
    let (dpsipl, depspl) = (dp * U2R, de * U2R);

    (dpsils + dpsipl, depsls + depspl)
}

/// Nutation in longitude and obliquity for the TT date `jd1 + jd2`, following `eraNut06a`.
///
/// Returns `(dpsi, deps)` in radians. This is IAU 2000A nutation with the IAU 2006
/// (P03) correction factors applied.
pub fn nut06a(jd1: f64, jd2: f64) -> (f64, f64) {
    let t = centuries_since_j2000(jd1, jd2);
    let fj2 = -2.7774e-6 * t;
    let (dp, de) = nut00a(t);
    (dp + dp * (0.4697e-6 + fj2), de + de * fj2)
}

/// The mean obliquity of the ecliptic in radians for the TT date `jd1 + jd2`, following `eraObl06`.
pub fn obl06(jd1: f64, jd2: f64) -> f64 {
    let t = centuries_since_j2000(jd1, jd2);
    let poly = 84381.406
        + (-46.836769
            + (-0.0001831 + (0.00200340 + (-0.000000576 + (-0.0000000434) * t) * t) * t) * t)
            * t;
    poly * DAS2R
}

/// Nutation in longitude and obliquity for each TT date `(jd1[i], jd2[i])`.
///
/// Returns the arrays `dpsi` and `deps` in radians. The dates are computed in parallel.
#[pyfunction(name = "nut06a")]
pub fn py_nut06a<'py>(
    py: Python<'py>,
    jd1: PyReadonlyArray1<'py, f64>,
    jd2: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Array<'py>, Array<'py>)> {
    let (first, second) = (jd1.as_slice()?, jd2.as_slice()?);
    check_same_len(first.len(), second.len())?;
    let pairs: Vec<(f64, f64)> = first
        .par_iter()
        .zip(second.par_iter())
        .map(|(&a, &b)| nut06a(a, b))
        .collect();
    let dpsi: Vec<f64> = pairs.iter().map(|pair| pair.0).collect();
    let deps: Vec<f64> = pairs.iter().map(|pair| pair.1).collect();
    Ok((PyArray1::from_vec(py, dpsi), PyArray1::from_vec(py, deps)))
}

/// The mean obliquity of the ecliptic in radians for each TT date `(jd1[i], jd2[i])`.
#[pyfunction(name = "obl06")]
pub fn py_obl06<'py>(
    py: Python<'py>,
    jd1: PyReadonlyArray1<'py, f64>,
    jd2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py>> {
    let (first, second) = (jd1.as_slice()?, jd2.as_slice()?);
    check_same_len(first.len(), second.len())?;
    let out: Vec<f64> = first
        .par_iter()
        .zip(second.par_iter())
        .map(|(&a, &b)| obl06(a, b))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At J2000.0 pyerfa 2.0.1 gives these values, in radians.
    #[test]
    fn nutation_at_j2000_matches_erfa() {
        let (dpsi, deps) = nut06a(DJ00, 0.0);
        assert!((dpsi - -6.754425598969513e-5).abs() < 1e-15);
        assert!((deps - -2.7970831192374137e-5).abs() < 1e-15);
    }

    #[test]
    fn mean_obliquity_at_j2000_is_84381_arcsec() {
        assert!((obl06(DJ00, 0.0) - 84381.406 * DAS2R).abs() < 1e-15);
    }

    #[test]
    fn the_split_of_the_date_does_not_change_the_result() {
        let whole = nut06a(2451545.5, 0.0);
        let split = nut06a(2451545.0, 0.5);
        assert!((whole.0 - split.0).abs() < 1e-16);
        assert!((whole.1 - split.1).abs() < 1e-16);
    }
}
