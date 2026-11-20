// This file reproduces routines from ERFA (Essential Routines for Fundamental
// Astronomy), version 2.0.1, from src/plan94.c in the liberfa/erfa
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
//! The planets' positions, from ERFA's `eraPlan94`.
//!
//! The tables of mean elements and periodic terms are copied from ERFA's C source,
//! with each number written as in the C source. The arithmetic keeps ERFA's order
//! and fuses multiplies and adds where the C build does.

/// Mean semi-major axes (au): the constant, linear and quadratic terms.
#[rustfmt::skip]
const MEAN_A: [[f64; 3]; 8] = [
    [0.3870983098, 0.0, 0.0],
    [0.7233298200, 0.0, 0.0],
    [1.0000010178, 0.0, 0.0],
    [1.5236793419, 3e-10, 0.0],
    [5.2026032092, 19132e-10, -39e-10],
    [9.5549091915, -0.0000213896, 444e-10],
    [19.2184460618, -3716e-10, 979e-10],
    [30.1103868694, -16635e-10, 686e-10],
];

/// Mean longitudes (degree and arcsecond terms).
#[rustfmt::skip]
const MEAN_DLM: [[f64; 3]; 8] = [
    [252.25090552, 5381016286.88982, -1.92789],
    [181.97980085, 2106641364.33548, 0.59381],
    [100.46645683, 1295977422.83429, -2.04411],
    [355.43299958, 689050774.93988, 0.94264],
    [34.35151874, 109256603.77991, -30.60378],
    [50.07744430, 43996098.55732, 75.61614],
    [314.05500511, 15424811.93933, -1.75083],
    [304.34866548, 7865503.20744, 0.21103],
];

/// Mean eccentricities.
#[rustfmt::skip]
const MEAN_E: [[f64; 3]; 8] = [
    [0.2056317526, 0.0002040653, -28349e-10],
    [0.0067719164, -0.0004776521, 98127e-10],
    [0.0167086342, -0.0004203654, -0.0000126734],
    [0.0934006477, 0.0009048438, -80641e-10],
    [0.0484979255, 0.0016322542, -0.0000471366],
    [0.0555481426, -0.0034664062, -0.0000643639],
    [0.0463812221, -0.0002729293, 0.0000078913],
    [0.0094557470, 0.0000603263, 0.0],
];

/// Longitudes of perihelion (degree and arcsecond terms).
#[rustfmt::skip]
const MEAN_PI: [[f64; 3]; 8] = [
    [77.45611904, 5719.11590, -4.83016],
    [131.56370300, 175.48640, -498.48184],
    [102.93734808, 11612.35290, 53.27577],
    [336.06023395, 15980.45908, -62.32800],
    [14.33120687, 7758.75163, 259.95938],
    [93.05723748, 20395.49439, 190.25952],
    [173.00529106, 3215.56238, -34.09288],
    [48.12027554, 1050.71912, 27.39717],
];

/// Inclinations (degree and arcsecond terms).
#[rustfmt::skip]
const MEAN_DINC: [[f64; 3]; 8] = [
    [7.00498625, -214.25629, 0.28977],
    [3.39466189, -30.84437, -11.67836],
    [0.0, 469.97289, -3.35053],
    [1.84972648, -293.31722, -8.11830],
    [1.30326698, -71.55890, 11.95297],
    [2.48887878, 91.85195, -17.66225],
    [0.77319689, -60.72723, 1.25759],
    [1.76995259, 8.12333, 0.08135],
];

/// Longitudes of the ascending node (degree and arcsecond terms).
#[rustfmt::skip]
const MEAN_OMEGA: [[f64; 3]; 8] = [
    [48.33089304, -4515.21727, -31.79892],
    [76.67992019, -10008.48154, -51.32614],
    [174.87317577, -8679.27034, 15.34191],
    [49.55809321, -10620.90088, -230.57416],
    [100.46440702, 6362.03561, 326.52178],
    [113.66550252, -9240.19942, -66.23743],
    [74.00595701, 2669.15033, 145.93964],
    [131.78405702, -221.94322, -0.78728],
];

/// Arguments of the semi-major axis terms, as multiples of the mean motion.
#[rustfmt::skip]
const KP: [[f64; 9]; 8] = [
    [
        69613.0, 75645.0, 88306.0, 59899.0, 15746.0, 71087.0, 142173.0, 3086.0, 0.0,
    ],
    [
        21863.0, 32794.0, 26934.0, 10931.0, 26250.0, 43725.0, 53867.0, 28939.0, 0.0,
    ],
    [
        16002.0, 21863.0, 32004.0, 10931.0, 14529.0, 16368.0, 15318.0, 32794.0, 0.0,
    ],
    [
        6345.0, 7818.0, 15636.0, 7077.0, 8184.0, 14163.0, 1107.0, 4872.0, 0.0,
    ],
    [
        1760.0, 1454.0, 1167.0, 880.0, 287.0, 2640.0, 19.0, 2047.0, 1454.0,
    ],
    [574.0, 0.0, 880.0, 287.0, 19.0, 1760.0, 1167.0, 306.0, 574.0],
    [204.0, 0.0, 177.0, 1265.0, 4.0, 385.0, 200.0, 208.0, 204.0],
    [0.0, 102.0, 106.0, 4.0, 98.0, 1367.0, 487.0, 204.0, 0.0],
];

/// Cosine amplitudes of the semi-major axis terms, in units of 1e-7 au.
#[rustfmt::skip]
const CA: [[f64; 9]; 8] = [
    [4.0, -13.0, 11.0, -9.0, -9.0, -3.0, -1.0, 4.0, 0.0],
    [-156.0, 59.0, -42.0, 6.0, 19.0, -20.0, -10.0, -12.0, 0.0],
    [64.0, -152.0, 62.0, -8.0, 32.0, -41.0, 19.0, -11.0, 0.0],
    [124.0, 621.0, -145.0, 208.0, 54.0, -57.0, 30.0, 15.0, 0.0],
    [
        -23437.0, -2634.0, 6601.0, 6259.0, -1507.0, -1821.0, 2620.0, -2115.0, -1489.0,
    ],
    [
        62911.0, -119919.0, 79336.0, 17814.0, -24241.0, 12068.0, 8306.0, -4893.0, 8902.0,
    ],
    [
        389061.0, -262125.0, -44088.0, 8387.0, -22976.0, -2093.0, -615.0, -9720.0, 6633.0,
    ],
    [
        -412235.0, -157046.0, -31430.0, 37817.0, -9740.0, -13.0, -7449.0, 9644.0, 0.0,
    ],
];

/// Sine amplitudes of the semi-major axis terms, in units of 1e-7 au.
#[rustfmt::skip]
const SA: [[f64; 9]; 8] = [
    [-29.0, -1.0, 9.0, 6.0, -6.0, 5.0, 4.0, 0.0, 0.0],
    [-48.0, -125.0, -26.0, -37.0, 18.0, -13.0, -20.0, -2.0, 0.0],
    [-150.0, -46.0, 68.0, 54.0, 14.0, 24.0, -28.0, 22.0, 0.0],
    [-621.0, 532.0, -694.0, -20.0, 192.0, -94.0, 71.0, -73.0, 0.0],
    [
        -14614.0, -19828.0, -5869.0, 1881.0, -4372.0, -2255.0, 782.0, 930.0, 913.0,
    ],
    [
        139737.0, 0.0, 24667.0, 51123.0, -5102.0, 7429.0, -4095.0, -1976.0, -9566.0,
    ],
    [
        -138081.0, 0.0, 37205.0, -49039.0, -41901.0, -33872.0, -27037.0, -12474.0, 18797.0,
    ],
    [
        0.0, 28492.0, 133236.0, 69654.0, 52322.0, -49577.0, -26430.0, -3593.0, 0.0,
    ],
];

/// Arguments of the mean longitude terms, as multiples of the mean motion.
#[rustfmt::skip]
const KQ: [[f64; 10]; 8] = [
    [
        3086.0, 15746.0, 69613.0, 59899.0, 75645.0, 88306.0, 12661.0, 2658.0, 0.0, 0.0,
    ],
    [
        21863.0, 32794.0, 10931.0, 73.0, 4387.0, 26934.0, 1473.0, 2157.0, 0.0, 0.0,
    ],
    [
        10.0, 16002.0, 21863.0, 10931.0, 1473.0, 32004.0, 4387.0, 73.0, 0.0, 0.0,
    ],
    [
        10.0, 6345.0, 7818.0, 1107.0, 15636.0, 7077.0, 8184.0, 532.0, 10.0, 0.0,
    ],
    [
        19.0, 1760.0, 1454.0, 287.0, 1167.0, 880.0, 574.0, 2640.0, 19.0, 1454.0,
    ],
    [
        19.0, 574.0, 287.0, 306.0, 1760.0, 12.0, 31.0, 38.0, 19.0, 574.0,
    ],
    [
        4.0, 204.0, 177.0, 8.0, 31.0, 200.0, 1265.0, 102.0, 4.0, 204.0,
    ],
    [
        4.0, 102.0, 106.0, 8.0, 98.0, 1367.0, 487.0, 204.0, 4.0, 102.0,
    ],
];

/// Cosine amplitudes of the mean longitude terms, in units of 1e-7 radian.
#[rustfmt::skip]
const CL: [[f64; 10]; 8] = [
    [21.0, -95.0, -157.0, 41.0, -5.0, 42.0, 23.0, 30.0, 0.0, 0.0],
    [
        -160.0, -313.0, -235.0, 60.0, -74.0, -76.0, -27.0, 34.0, 0.0, 0.0,
    ],
    [
        -325.0, -322.0, -79.0, 232.0, -52.0, 97.0, 55.0, -41.0, 0.0, 0.0,
    ],
    [
        2268.0, -979.0, 802.0, 602.0, -668.0, -33.0, 345.0, 201.0, -55.0, 0.0,
    ],
    [
        7610.0, -4997.0, -7689.0, -5841.0, -2617.0, 1115.0, -748.0, -607.0, 6074.0, 354.0,
    ],
    [
        -18549.0, 30125.0, 20012.0, -730.0, 824.0, 23.0, 1289.0, -352.0, -14767.0, -2062.0,
    ],
    [
        -135245.0, -14594.0, 4197.0, -4030.0, -5630.0, -2898.0, 2540.0, -306.0, 2939.0, 1986.0,
    ],
    [
        89948.0, 2103.0, 8963.0, 2695.0, 3682.0, 1648.0, 866.0, -154.0, -1963.0, -283.0,
    ],
];

/// Sine amplitudes of the mean longitude terms, in units of 1e-7 radian.
#[rustfmt::skip]
const SL: [[f64; 10]; 8] = [
    [
        -342.0, 136.0, -23.0, 62.0, 66.0, -52.0, -33.0, 17.0, 0.0, 0.0,
    ],
    [
        524.0, -149.0, -35.0, 117.0, 151.0, 122.0, -71.0, -62.0, 0.0, 0.0,
    ],
    [
        -105.0, -137.0, 258.0, 35.0, -116.0, -88.0, -112.0, -80.0, 0.0, 0.0,
    ],
    [
        854.0, -205.0, -936.0, -240.0, 140.0, -341.0, -97.0, -232.0, 536.0, 0.0,
    ],
    [
        -56980.0, 8016.0, 1012.0, 1448.0, -3024.0, -3710.0, 318.0, 503.0, 3767.0, 577.0,
    ],
    [
        138606.0, -13478.0, -4964.0, 1441.0, -1319.0, -1482.0, 427.0, 1236.0, -9167.0, -1918.0,
    ],
    [
        71234.0, -41116.0, 5334.0, -4935.0, -1848.0, 66.0, 434.0, -1748.0, 3780.0, -701.0,
    ],
    [
        -47645.0, 11647.0, 2166.0, 3194.0, 679.0, 0.0, -244.0, -419.0, -2531.0, 48.0,
    ],
];

/// Julian date of the epoch J2000.0.
#[rustfmt::skip]
const DJ00: f64 = 2_451_545.0;
/// Days in a Julian millennium.
#[rustfmt::skip]
const DJM: f64 = 365_250.0;
/// Radians per arcsecond.
#[rustfmt::skip]
const DAS2R: f64 = 4.84813681109536e-06;
/// Two pi, in radians.
#[rustfmt::skip]
const D2PI: f64 = std::f64::consts::TAU;
/// Pi, in radians.
#[rustfmt::skip]
const DPI: f64 = std::f64::consts::PI;
/// Sine of the J2000.0 mean obliquity (IAU 1976).
#[rustfmt::skip]
const SINEPS: f64 = 0.3977771559319137;
/// Cosine of the J2000.0 mean obliquity (IAU 1976).
#[rustfmt::skip]
const COSEPS: f64 = 0.9174820620691818;
/// The maximum number of iterations allowed to solve Kepler's equation.
#[rustfmt::skip]
const KMAX: usize = 10;

/// The position of a planet from `eraPlan94`, in astronomical units and in the ICRS axes.
pub struct Plan94 {
    /// The heliocentric position of the planet.
    pub position: [f64; 3],
    /// ERFA's status: 0 for dates within a millennium of J2000.0, 1 outside that range,
    /// and 2 when Kepler's equation was not solved within the iteration limit.
    pub status: i32,
}

/// Normalises an angle to the range from minus pi to pi, as `eraAnpm` does.
fn anpm(a: f64) -> f64 {
    let mut w = a % D2PI;
    if w.abs() >= DPI {
        w -= if a < 0.0 { -D2PI } else { D2PI };
    }
    w
}

/// The heliocentric position of planet `np`, from `eraPlan94` (ERFA 2.0.1, from SOFA).
///
/// `np` is ERFA's planet number, 1 for Mercury to 8 for Neptune, with 3 for the
/// Earth-Moon barycentre. The velocity that ERFA also returns is not computed,
/// because Heliox does not use it.
///
/// # Panics
///
/// Panics if `np` is not in the range 1 to 8. The Python caller only passes valid numbers.
pub fn plan94(date1: f64, date2: f64, np: usize) -> Plan94 {
    assert!(
        (1..=8).contains(&np),
        "the planet number must be from 1 to 8"
    );
    let i = np - 1;
    let t = ((date1 - DJ00) + date2) / DJM;
    let mut status = if t.abs() <= 1.0 { 0 } else { 1 };

    // Mean elements, limited to t squared terms. The C build fuses each
    // multiply and add, so the polynomials use mul_add.
    let mut da = MEAN_A[i][2]
        .mul_add(t, MEAN_A[i][1])
        .mul_add(t, MEAN_A[i][0]);
    let mut dl = (3600.0_f64.mul_add(
        MEAN_DLM[i][0],
        MEAN_DLM[i][2].mul_add(t, MEAN_DLM[i][1]) * t,
    )) * DAS2R;
    let de = MEAN_E[i][2]
        .mul_add(t, MEAN_E[i][1])
        .mul_add(t, MEAN_E[i][0]);
    let dp = anpm(
        3600.0_f64.mul_add(MEAN_PI[i][0], MEAN_PI[i][2].mul_add(t, MEAN_PI[i][1]) * t) * DAS2R,
    );
    let di = (3600.0_f64.mul_add(
        MEAN_DINC[i][0],
        MEAN_DINC[i][2].mul_add(t, MEAN_DINC[i][1]) * t,
    )) * DAS2R;
    let dom = anpm(
        3600.0_f64.mul_add(
            MEAN_OMEGA[i][0],
            MEAN_OMEGA[i][2].mul_add(t, MEAN_OMEGA[i][1]) * t,
        ) * DAS2R,
    );

    // Trigonometric terms.
    let dmu = 0.35953620 * t;
    for k in 0..8 {
        let arga = KP[i][k] * dmu;
        let argl = KQ[i][k] * dmu;
        da = CA[i][k]
            .mul_add(arga.cos(), SA[i][k] * arga.sin())
            .mul_add(1e-7, da);
        dl = CL[i][k]
            .mul_add(argl.cos(), SL[i][k] * argl.sin())
            .mul_add(1e-7, dl);
    }
    let arga = KP[i][8] * dmu;
    da = (t * CA[i][8].mul_add(arga.cos(), SA[i][8] * arga.sin())).mul_add(1e-7, da);
    for k in 8..10 {
        let argl = KQ[i][k] * dmu;
        dl = (t * CL[i][k].mul_add(argl.cos(), SL[i][k] * argl.sin())).mul_add(1e-7, dl);
    }
    dl %= D2PI;

    // Solve Kepler's equation for the eccentric anomaly.
    let am = dl - dp;
    let mut ae = de.mul_add(am.sin(), am);
    let mut k = 0;
    let mut dae: f64 = 1.0;
    while k < KMAX && dae.abs() > 1e-12 {
        dae = de.mul_add(ae.sin(), am - ae) / (-de).mul_add(ae.cos(), 1.0);
        ae += dae;
        k += 1;
        if k == KMAX - 1 {
            status = 2;
        }
    }

    // True anomaly and distance.
    let ae2 = ae / 2.0;
    let at = 2.0 * (((1.0 + de) / (1.0 - de)).sqrt() * ae2.sin()).atan2(ae2.cos());
    let r = da * (-de).mul_add(ae.cos(), 1.0);

    // Position in the J2000.0 ecliptic, then rotated to the equator.
    let si2 = (di / 2.0).sin();
    let xq = si2 * dom.cos();
    let xp = si2 * dom.sin();
    let tl = at + dp;
    let xsw = tl.sin();
    let xcw = tl.cos();
    let xm2: f64 = 2.0 * xp.mul_add(xcw, -(xq * xsw));
    let ci2 = (di / 2.0).cos();
    let x = r * (-xm2).mul_add(xp, xcw);
    let y = r * xm2.mul_add(xq, xsw);
    let z = r * (-xm2 * ci2);

    Plan94 {
        position: [
            x,
            y.mul_add(COSEPS, -(z * SINEPS)),
            y.mul_add(SINEPS, z * COSEPS),
        ],
        status,
    }
}
