// This file reproduces routines from ERFA (Essential Routines for Fundamental
// Astronomy), version 2.0.1, from src/pfw06.c, src/ir.c, src/rz.c, src/rx.c and src/rxp.c in the liberfa/erfa
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
//! The precession angles and rotations that `eraMoon98` uses to take the Moon into
//! the GCRS, from ERFA's `eraPfw06`, `eraRz`, `eraRx` and `eraRxp`. The products
//! are fused where the C build fuses them, so the results match astropy exactly.

/// Julian date of the epoch J2000.0.
const DJ00: f64 = 2_451_545.0;
/// Days in a Julian century.
const DJC: f64 = 36_525.0;
/// Radians per arcsecond.
const DAS2R: f64 = 4.84813681109536e-06;

/// The bias and precession angles of the Fukushima-Williams form, in arcseconds,
/// as polynomials in Julian centuries from J2000.0. From `eraPfw06`, which
/// gives `gamb`, `phib` and `psib` in that order, in addition to the obliquity.
const GAMB: [f64; 6] = [
    -0.052928,
    10.556378,
    0.4932044,
    -0.00031238,
    -0.000002788,
    0.0000000260,
];
const PHIB: [f64; 6] = [
    84381.412819,
    -46.811016,
    0.0511268,
    0.00053289,
    -0.000000440,
    -0.0000000176,
];
const PSIB: [f64; 6] = [
    -0.041775,
    5038.481484,
    1.5584175,
    -0.00018522,
    -0.000026452,
    -0.0000000148,
];

/// A 3 by 3 rotation matrix, indexed as `m[row][column]`.
pub(super) type Matrix = [[f64; 3]; 3];

/// The identity matrix, the starting point of the rotations.
pub(super) const IDENTITY: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Evaluates a polynomial with coefficients `c` (lowest power first) at `t`,
/// in the nested form that ERFA uses: `c0 + (c1 + (c2 + ...) * t) * t`.
pub(super) fn horner(c: &[f64], t: f64) -> f64 {
    let (last, rest) = c
        .split_last()
        .expect("a polynomial has at least one coefficient");
    rest.iter().rev().fold(*last, |acc, &ck| acc.mul_add(t, ck))
}

/// The Fukushima-Williams bias and precession angles `(gamb, phib, psib)`, in radians.
///
/// This reproduces the three angles of `eraPfw06` (ERFA 2.0.1, from SOFA) that
/// `eraMoon98` uses. `date1` and `date2` form a TT Julian date.
pub(super) fn fukushima_williams(date1: f64, date2: f64) -> (f64, f64, f64) {
    let t = ((date1 - DJ00) + date2) / DJC;
    (
        horner(&GAMB, t) * DAS2R,
        horner(&PHIB, t) * DAS2R,
        horner(&PSIB, t) * DAS2R,
    )
}

/// Rotates the matrix `r` about the z-axis by `psi`, as `eraRz` does.
/// Rotates the matrix `r` about the z-axis by `psi`, as `eraRz` does.
pub(super) fn rz(psi: f64, r: &mut Matrix) {
    let s = psi.sin();
    let c = psi.cos();
    let a00 = c.mul_add(r[0][0], s * r[1][0]);
    let a01 = c.mul_add(r[0][1], s * r[1][1]);
    let a02 = c.mul_add(r[0][2], s * r[1][2]);
    let a10 = (-s).mul_add(r[0][0], c * r[1][0]);
    let a11 = (-s).mul_add(r[0][1], c * r[1][1]);
    let a12 = (-s).mul_add(r[0][2], c * r[1][2]);
    r[0] = [a00, a01, a02];
    r[1] = [a10, a11, a12];
}

/// Rotates the matrix `r` about the x-axis by `phi`, as `eraRx` does.
/// Rotates the matrix `r` about the x-axis by `phi`, as `eraRx` does.
pub(super) fn rx(phi: f64, r: &mut Matrix) {
    let s = phi.sin();
    let c = phi.cos();
    let a10 = c.mul_add(r[1][0], s * r[2][0]);
    let a11 = c.mul_add(r[1][1], s * r[2][1]);
    let a12 = c.mul_add(r[1][2], s * r[2][2]);
    let a20 = (-s).mul_add(r[1][0], c * r[2][0]);
    let a21 = (-s).mul_add(r[1][1], c * r[2][1]);
    let a22 = (-s).mul_add(r[1][2], c * r[2][2]);
    r[1] = [a10, a11, a12];
    r[2] = [a20, a21, a22];
}

/// The product of the matrix `r` and the vector `p`, as `eraRxp` computes it.
/// The product of the matrix `r` and the vector `p`, as `eraRxp` computes it.
pub(super) fn rxp(r: &Matrix, p: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (j, row) in r.iter().enumerate() {
        let mut w = 0.0;
        for i in 0..3 {
            w = row[i].mul_add(p[i], w);
        }
        out[j] = w;
    }
    out
}
