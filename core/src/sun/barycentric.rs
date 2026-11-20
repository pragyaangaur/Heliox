// This file reproduces routines from ERFA (Essential Routines for Fundamental
// Astronomy), version 2.0.1, from src/pvppv.c and src/pvmpv.c, and the way astropy combines them in the liberfa/erfa
// repository. The notice below is ERFA's licence, reproduced in full as the
// licence requires.
//
//
//
//   Copyright (C) 2013-2023, NumFOCUS Foundation.
//   All rights reserved.
//
//   This library is derived, with permission, from the International
//   Astronomical Union's "Standards of Fundamental Astronomy" library,
//   available from http://www.iausofa.org.
//
//   The ERFA version is intended to retain identical functionality to
//   the SOFA library, but made distinct through different function and
//   file names, as set out in the SOFA license conditions.  The SOFA
//   original has a role as a reference standard for the IAU and IERS,
//   and consequently redistribution is permitted only in its unaltered
//   state.  The ERFA version is not subject to this restriction and
//   therefore can be included in distributions which do not support the
//   concept of "read only" software.
//
//   Although the intent is to replicate the SOFA API (other than
//   replacement of prefix names) and results (with the exception of
//   bugs;  any that are discovered will be fixed), SOFA is not
//   responsible for any errors found in this version of the library.
//
//   If you wish to acknowledge the SOFA heritage, please acknowledge
//   that you are using a library derived from SOFA, rather than SOFA
//   itself.
//
//
//   TERMS AND CONDITIONS
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
//
//   1 Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   2 Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//
//   3 Neither the name of the Standards Of Fundamental Astronomy Board,
//     the International Astronomical Union nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//   FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
//   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//   POSSIBILITY OF SUCH DAMAGE.
//
//!
//! Barycentric positions of the Sun, the Earth, the Moon and the planets, for
//! astropy's built-in ephemeris.
//!
//! This is the calculation in astropy's `_get_body_barycentric_posvel` for the
//! `builtin` ephemeris. It combines ERFA's `eraEpv00`, `eraMoon98` and
//! `eraPlan94` with ERFA's `eraPvppv` and `eraPvmpv` additions.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::epv00::epv00;
use super::moon98::moon98;
use super::plan94::plan94;

/// A body that astropy's built-in ephemeris can place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    Earth,
    Sun,
    Moon,
    /// A planet, or the Earth-Moon barycentre, with its ERFA planet number from 1 to 8.
    Planet(usize),
}

impl Body {
    /// The body with this lower-case name, as `astropy.coordinates` spells it.
    pub fn from_name(name: &str) -> Option<Body> {
        Some(match name {
            "earth" => Body::Earth,
            "sun" => Body::Sun,
            "moon" => Body::Moon,
            "mercury" => Body::Planet(1),
            "venus" => Body::Planet(2),
            "earth-moon-barycenter" => Body::Planet(3),
            "mars" => Body::Planet(4),
            "jupiter" => Body::Planet(5),
            "saturn" => Body::Planet(6),
            "uranus" => Body::Planet(7),
            "neptune" => Body::Planet(8),
            _ => return None,
        })
    }
}

/// A barycentric position, with the ERFA status of the calculation.
pub struct Barycentric {
    /// The position in astronomical units, in the ICRS axes.
    pub position: [f64; 3],
    /// The largest ERFA status used: 0 means the date is inside the range the models cover.
    pub status: i32,
}

/// `eraPvmpv` for positions: the difference of two vectors.
fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `eraPvppv` for positions: the sum of two vectors.
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The barycentric position of `body` at the TDB Julian date `date1 + date2`.
///
/// This follows `_get_body_barycentric_posvel` in astropy for the built-in
/// ephemeris. The Sun is the barycentric Earth less the heliocentric Earth. The
/// Moon is its geocentric position plus the barycentric Earth, and a planet is
/// its heliocentric position plus the barycentric Sun.
pub fn barycentric(body: Body, date1: f64, date2: f64) -> Barycentric {
    let earth = epv00(date1, date2);
    let sun = subtract(earth.barycentric, earth.heliocentric);
    let (position, status) = match body {
        Body::Earth => (earth.barycentric, earth.status),
        Body::Sun => (sun, earth.status),
        Body::Moon => (add(moon98(date1, date2), earth.barycentric), earth.status),
        Body::Planet(np) => {
            let planet = plan94(date1, date2, np);
            (add(planet.position, sun), earth.status.max(planet.status))
        }
    };
    Barycentric { position, status }
}

/// The barycentric positions of a body at many TDB Julian dates.
///
/// Returns the x components of every date, then the y components, then the z
/// components, as one flat array of length `3n` for `n` dates. Also returns the
/// largest ERFA status over all the dates, so that the caller can fall back to
/// astropy, which issues ERFA's warnings, whenever it is not zero.
///
/// # Errors
///
/// Raises `ValueError` if `name` is not a body of the built-in ephemeris, or if
/// the two date arrays differ in length.
#[pyfunction]
pub fn body_barycentric<'py>(
    py: Python<'py>,
    name: &str,
    jd1: PyReadonlyArray1<'py, f64>,
    jd2: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, i32)> {
    let body = Body::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!("{name} is not a body of the built-in ephemeris"))
    })?;
    let (first, second) = (jd1.as_slice()?, jd2.as_slice()?);
    if first.len() != second.len() {
        return Err(PyValueError::new_err(
            "the input arrays must have the same length",
        ));
    }

    let n = first.len();
    let mut out = vec![0.0; 3 * n];
    let mut status = 0;
    for (k, (&a, &b)) in first.iter().zip(second).enumerate() {
        let result = barycentric(body, a, b);
        status = status.max(result.status);
        for (axis, value) in result.position.into_iter().enumerate() {
            out[axis * n + k] = value;
        }
    }
    Ok((PyArray1::from_vec(py, out), status))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The length of a vector.
    fn norm(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn names_map_to_bodies() {
        assert_eq!(Body::from_name("earth"), Some(Body::Earth));
        assert_eq!(
            Body::from_name("earth-moon-barycenter"),
            Some(Body::Planet(3))
        );
        assert_eq!(Body::from_name("pluto"), None);
    }

    #[test]
    fn earth_is_about_one_au_from_the_sun_in_january() {
        // 2000 January 1.5 TDB, when the Earth is near perihelion.
        let sun = barycentric(Body::Sun, 2_451_544.5, 0.0);
        let earth = barycentric(Body::Earth, 2_451_544.5, 0.0);
        let separation = norm(subtract(earth.position, sun.position));
        assert!((separation - 0.983).abs() < 0.002, "{separation}");
        assert_eq!(earth.status, 0);
    }

    #[test]
    fn the_moon_is_between_perigee_and_apogee_from_the_earth() {
        let earth = barycentric(Body::Earth, 2_451_545.0, 0.0);
        let moon = barycentric(Body::Moon, 2_451_545.0, 0.0);
        let distance = norm(subtract(moon.position, earth.position));
        assert!((0.00239..0.00275).contains(&distance), "{distance}");
    }

    #[test]
    fn dates_far_from_2000_set_a_status() {
        assert_eq!(
            barycentric(Body::Earth, 2_451_545.0 + 100.0 * 365.25 * 2.0, 0.0).status,
            1
        );
    }
}
