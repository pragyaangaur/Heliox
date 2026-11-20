// This file reproduces routines from ERFA (Essential Routines for Fundamental
// Astronomy), version 2.0.1, from src/moon98.c in the liberfa/erfa
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
//! The Moon's geocentric position, from ERFA's `eraMoon98`.
//!
//! The periodic series are copied from ERFA's C source. The position is rotated
//! into the GCRS with the same Fukushima-Williams matrix that ERFA uses, and the
//! series are summed in the same order.

use super::precession::{fukushima_williams, horner, rx, rxp, rz, IDENTITY};

/// Longitude and distance series: (D, M, M', F multiples, sine coefficient of longitude in degrees, cosine coefficient of distance in metres).
#[rustfmt::skip]
const LONGITUDE_TERMS: [(i32, i32, i32, i32, f64, f64); 60] = [
    (0, 0, 1, 0, 6.288774, -20905355.0),
    (2, 0, -1, 0, 1.274027, -3699111.0),
    (2, 0, 0, 0, 0.658314, -2955968.0),
    (0, 0, 2, 0, 0.213618, -569925.0),
    (0, 1, 0, 0, -0.185116, 48888.0),
    (0, 0, 0, 2, -0.114332, -3149.0),
    (2, 0, -2, 0, 0.058793, 246158.0),
    (2, -1, -1, 0, 0.057066, -152138.0),
    (2, 0, 1, 0, 0.053322, -170733.0),
    (2, -1, 0, 0, 0.045758, -204586.0),
    (0, 1, -1, 0, -0.040923, -129620.0),
    (1, 0, 0, 0, -0.034720, 108743.0),
    (0, 1, 1, 0, -0.030383, 104755.0),
    (2, 0, 0, -2, 0.015327, 10321.0),
    (0, 0, 1, 2, -0.012528, 0.0),
    (0, 0, 1, -2, 0.010980, 79661.0),
    (4, 0, -1, 0, 0.010675, -34782.0),
    (0, 0, 3, 0, 0.010034, -23210.0),
    (4, 0, -2, 0, 0.008548, -21636.0),
    (2, 1, -1, 0, -0.007888, 24208.0),
    (2, 1, 0, 0, -0.006766, 30824.0),
    (1, 0, -1, 0, -0.005163, -8379.0),
    (1, 1, 0, 0, 0.004987, -16675.0),
    (2, -1, 1, 0, 0.004036, -12831.0),
    (2, 0, 2, 0, 0.003994, -10445.0),
    (4, 0, 0, 0, 0.003861, -11650.0),
    (2, 0, -3, 0, 0.003665, 14403.0),
    (0, 1, -2, 0, -0.002689, -7003.0),
    (2, 0, -1, 2, -0.002602, 0.0),
    (2, -1, -2, 0, 0.002390, 10056.0),
    (1, 0, 1, 0, -0.002348, 6322.0),
    (2, -2, 0, 0, 0.002236, -9884.0),
    (0, 1, 2, 0, -0.002120, 5751.0),
    (0, 2, 0, 0, -0.002069, 0.0),
    (2, -2, -1, 0, 0.002048, -4950.0),
    (2, 0, 1, -2, -0.001773, 4130.0),
    (2, 0, 0, 2, -0.001595, 0.0),
    (4, -1, -1, 0, 0.001215, -3958.0),
    (0, 0, 2, 2, -0.001110, 0.0),
    (3, 0, -1, 0, -0.000892, 3258.0),
    (2, 1, 1, 0, -0.000810, 2616.0),
    (4, -1, -2, 0, 0.000759, -1897.0),
    (0, 2, -1, 0, -0.000713, -2117.0),
    (2, 2, -1, 0, -0.000700, 2354.0),
    (2, 1, -2, 0, 0.000691, 0.0),
    (2, -1, 0, -2, 0.000596, 0.0),
    (4, 0, 1, 0, 0.000549, -1423.0),
    (0, 0, 4, 0, 0.000537, -1117.0),
    (4, -1, 0, 0, 0.000520, -1571.0),
    (1, 0, -2, 0, -0.000487, -1739.0),
    (2, 1, 0, -2, -0.000399, 0.0),
    (0, 0, 2, -2, -0.000381, -4421.0),
    (1, 1, 1, 0, 0.000351, 0.0),
    (3, 0, -2, 0, -0.000340, 0.0),
    (4, 0, -3, 0, 0.000330, 0.0),
    (2, -1, 2, 0, 0.000327, 0.0),
    (0, 2, 1, 0, -0.000323, 1165.0),
    (1, 1, -1, 0, 0.000299, 0.0),
    (2, 0, 3, 0, 0.000294, 0.0),
    (2, 0, -1, -2, 0.000000, 8752.0),
];

/// Latitude series: (D, M, M', F multiples, sine coefficient of latitude in degrees).
#[rustfmt::skip]
const LATITUDE_TERMS: [(i32, i32, i32, i32, f64); 60] = [
    (0, 0, 0, 1, 5.128122),
    (0, 0, 1, 1, 0.280602),
    (0, 0, 1, -1, 0.277693),
    (2, 0, 0, -1, 0.173237),
    (2, 0, -1, 1, 0.055413),
    (2, 0, -1, -1, 0.046271),
    (2, 0, 0, 1, 0.032573),
    (0, 0, 2, 1, 0.017198),
    (2, 0, 1, -1, 0.009266),
    (0, 0, 2, -1, 0.008822),
    (2, -1, 0, -1, 0.008216),
    (2, 0, -2, -1, 0.004324),
    (2, 0, 1, 1, 0.004200),
    (2, 1, 0, -1, -0.003359),
    (2, -1, -1, 1, 0.002463),
    (2, -1, 0, 1, 0.002211),
    (2, -1, -1, -1, 0.002065),
    (0, 1, -1, -1, -0.001870),
    (4, 0, -1, -1, 0.001828),
    (0, 1, 0, 1, -0.001794),
    (0, 0, 0, 3, -0.001749),
    (0, 1, -1, 1, -0.001565),
    (1, 0, 0, 1, -0.001491),
    (0, 1, 1, 1, -0.001475),
    (0, 1, 1, -1, -0.001410),
    (0, 1, 0, -1, -0.001344),
    (1, 0, 0, -1, -0.001335),
    (0, 0, 3, 1, 0.001107),
    (4, 0, 0, -1, 0.001021),
    (4, 0, -1, 1, 0.000833),
    (0, 0, 1, -3, 0.000777),
    (4, 0, -2, 1, 0.000671),
    (2, 0, 0, -3, 0.000607),
    (2, 0, 2, -1, 0.000596),
    (2, -1, 1, -1, 0.000491),
    (2, 0, -2, 1, -0.000451),
    (0, 0, 3, -1, 0.000439),
    (2, 0, 2, 1, 0.000422),
    (2, 0, -3, -1, 0.000421),
    (2, 1, -1, 1, -0.000366),
    (2, 1, 0, 1, -0.000351),
    (4, 0, 0, 1, 0.000331),
    (2, -1, 1, 1, 0.000315),
    (2, -2, 0, -1, 0.000302),
    (0, 0, 1, 3, -0.000283),
    (2, 1, 1, -1, -0.000229),
    (1, 1, 0, -1, 0.000223),
    (1, 1, 0, 1, 0.000223),
    (0, 1, -2, -1, -0.000220),
    (2, 1, -1, -1, -0.000220),
    (1, 0, 1, 1, -0.000185),
    (2, -1, -2, -1, 0.000181),
    (0, 1, 2, 1, -0.000177),
    (4, 0, -2, -1, 0.000176),
    (4, -1, -1, -1, 0.000166),
    (1, 0, 1, -1, -0.000164),
    (4, 0, 1, -1, 0.000132),
    (1, 0, -1, -1, -0.000119),
    (4, -1, 0, -1, 0.000115),
    (2, -2, 0, 1, 0.000107),
];

/// Julian date of the epoch J2000.0.
#[rustfmt::skip]
const DJ00: f64 = 2_451_545.0;
/// Days in a Julian century.
#[rustfmt::skip]
const DJC: f64 = 36_525.0;
/// Radians per degree.
#[rustfmt::skip]
const DD2R: f64 = 1.745_329_251_994_329_5e-2;
/// Astronomical unit, in metres.
#[rustfmt::skip]
const DAU: f64 = 149_597_870.7e3;

/// Moon's mean longitude, polynomial in centuries (degrees).
#[rustfmt::skip]
const ELP: [f64; 5] = [
    218.31665436,
    481267.88123421,
    -0.0015786,
    1.0 / 538841.0,
    -1.0 / 65194000.0,
];
/// Moon's mean elongation (degrees).
#[rustfmt::skip]
const D: [f64; 5] = [
    297.8501921,
    445267.1114034,
    -0.0018819,
    1.0 / 545868.0,
    1.0 / 113065000.0,
];
/// Sun's mean anomaly (degrees).
#[rustfmt::skip]
const EM: [f64; 5] = [
    357.5291092,
    35999.0502909,
    -0.0001536,
    1.0 / 24490000.0,
    0.0,
];
/// Moon's mean anomaly (degrees).
#[rustfmt::skip]
const EMP: [f64; 5] = [
    134.9633964,
    477198.8675055,
    0.0087414,
    1.0 / 69699.0,
    -1.0 / 14712000.0,
];
/// Mean distance of the Moon from its ascending node (degrees).
#[rustfmt::skip]
const F: [f64; 5] = [
    93.2720950,
    483202.0175233,
    -0.0036539,
    1.0 / 3526000.0,
    1.0 / 863310000.0,
];

/// Meeus' A1, due to Venus: constant and rate (degrees).
#[rustfmt::skip]
const A10: f64 = 119.75;
#[rustfmt::skip]
const A11: f64 = 131.849;
/// Meeus' A2, due to Jupiter: constant and rate (degrees).
#[rustfmt::skip]
const A20: f64 = 53.09;
#[rustfmt::skip]
const A21: f64 = 479264.290;
/// Meeus' A3, due to the sidereal motion of the Moon in longitude: constant and rate (degrees).
#[rustfmt::skip]
const A30: f64 = 313.45;
#[rustfmt::skip]
const A31: f64 = 481266.484;

/// Coefficients of Meeus' additive terms in longitude (degrees).
#[rustfmt::skip]
const AL1: f64 = 0.003958;
#[rustfmt::skip]
const AL2: f64 = 0.001962;
#[rustfmt::skip]
const AL3: f64 = 0.000318;
/// Coefficients of Meeus' additive terms in latitude (degrees).
#[rustfmt::skip]
const AB1: f64 = -0.002235;
#[rustfmt::skip]
const AB2: f64 = 0.000382;
#[rustfmt::skip]
const AB3: f64 = 0.000175;
#[rustfmt::skip]
const AB4: f64 = 0.000175;
#[rustfmt::skip]
const AB5: f64 = 0.000127;
#[rustfmt::skip]
const AB6: f64 = -0.000115;

/// The fixed term in the distance of the Moon, in metres.
#[rustfmt::skip]
const R0: f64 = 385_000_560.0;
/// Coefficients of the eccentricity factor E.
#[rustfmt::skip]
const E1: f64 = -0.002516;
#[rustfmt::skip]
const E2: f64 = -0.0000074;

/// The geocentric position of the Moon from `eraMoon98` (ERFA 2.0.1, from SOFA), in astronomical units and GCRS axes.
///
/// `date1` and `date2` form a TT Julian date. ERFA's note says TDB gives
/// the same result to well within the accuracy of the algorithm, and
/// astropy passes TDB, so this module does the same. The velocity that ERFA
/// also returns is not computed, because Heliox does not use it.
pub fn moon98(date1: f64, date2: f64) -> [f64; 3] {
    let t = ((date1 - DJ00) + date2) / DJC;

    // Fundamental arguments, in radians.
    let elp = DD2R * (horner(&ELP, t) % 360.0);
    let d = DD2R * (horner(&D, t) % 360.0);
    let em = DD2R * (horner(&EM, t) % 360.0);
    let emp = DD2R * (horner(&EMP, t) % 360.0);
    let f = DD2R * (horner(&F, t) % 360.0);
    let a1 = DD2R * A11.mul_add(t, A10);
    let a2 = DD2R * A21.mul_add(t, A20);
    let a3 = DD2R * A31.mul_add(t, A30);
    let e = E2.mul_add(t, E1).mul_add(t, 1.0);
    let args = Fundamental {
        d,
        em,
        emp,
        f,
        e,
        esq: e * e,
    };

    // Meeus' additive terms start off the sums. The C build fuses each
    // multiply and add, so these use mul_add, summed from the first term.
    let elpmf = elp - f;
    let mut vel = AL3.mul_add(a2.sin(), AL1.mul_add(a1.sin(), AL2 * elpmf.sin()));
    let mut vr = 0.0;
    let a1mf = a1 - f;
    let a1pf = a1 + f;
    let dlpmp = elp - emp;
    let slpmp = elp + emp;
    let mut vb = AB6.mul_add(
        slpmp.sin(),
        AB5.mul_add(
            dlpmp.sin(),
            AB4.mul_add(
                a1pf.sin(),
                AB3.mul_add(a1mf.sin(), AB1.mul_add(elp.sin(), AB2 * a3.sin())),
            ),
        ),
    );

    // The periodic series, summed from the last term to the first as ERFA does.
    for &(nd, nem, nemp, nf, coefl, coefr) in LONGITUDE_TERMS.iter().rev() {
        let (arg, en) = periodic_term(nd, nem, nemp, nf, &args);
        vel = coefl.mul_add(arg.sin() * en, vel);
        vr = coefr.mul_add(arg.cos() * en, vr);
    }
    for &(nd, nem, nemp, nf, coefb) in LATITUDE_TERMS.iter().rev() {
        let (arg, en) = periodic_term(nd, nem, nemp, nf, &args);
        vb = coefb.mul_add(arg.sin() * en, vb);
    }

    let el = DD2R.mul_add(vel, elp);
    let r = (vr + R0) / DAU;
    let b = vb * DD2R;

    // Longitude, latitude and distance to Cartesian coordinates, as eraS2pv does.
    let rcp = r * b.cos();
    let position = [rcp * el.cos(), rcp * el.sin(), r * b.sin()];

    // Mean ecliptic of date to GCRS, through the Fukushima-Williams angles.
    let (gamb, phib, psib) = fukushima_williams(date1, date2);
    let mut rm = IDENTITY;
    rz(psib, &mut rm);
    rx(-phib, &mut rm);
    rz(-gamb, &mut rm);
    rxp(&rm, position)
}

/// The fundamental arguments and the eccentricity factors that a periodic term needs.
struct Fundamental {
    d: f64,
    em: f64,
    emp: f64,
    f: f64,
    e: f64,
    esq: f64,
}

/// The argument of a periodic term, and the eccentricity factor that multiplies it.
fn periodic_term(nd: i32, nem: i32, nemp: i32, nf: i32, args: &Fundamental) -> (f64, f64) {
    let arg = f64::from(nf).mul_add(
        args.f,
        f64::from(nemp).mul_add(
            args.emp,
            f64::from(nd).mul_add(args.d, f64::from(nem) * args.em),
        ),
    );
    let en = match nem.abs() {
        1 => args.e,
        2 => args.esq,
        _ => 1.0,
    };
    (arg, en)
}
