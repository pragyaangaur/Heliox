//! The per pixel coordinate chain of `sunpy.physics.differential_rotation._warp_sun_coordinates`
//! (sunpy 8.0.1).
//!
//! For every output pixel the original does this, one astropy step after another on whole arrays:
//!
//! 1. pixel to helioprojective angles `(Tx, Ty)` with the map WCS (TAN projection),
//! 2. relabel them as seen by the new observer and find the point on the solar surface (the near
//!    solution of the law of cosines, NaN off the disk),
//! 3. helioprojective to heliocentric to heliographic Stonyhurst, a rotation `A`,
//! 4. subtract the differential rotation, which only depends on the latitude,
//! 5. heliographic at the new observer time to heliocentric as seen by the map observer, a
//!    rotation `B` (the transformation under `transform_with_sun_center`, so no translation),
//!    and mark pixels with a negative heliocentric `z` as NaN,
//! 6. heliocentric to helioprojective angles, and the WCS back to a source pixel.
//!
//! Everything that depends only on the observers and the time interval is computed by the Python
//! layer with sunpy and astropy and passed in as plain numbers (the WCS parameters, the two
//! rotation matrices, the observer distances and the three rotation rate coefficients). This
//! module does the arithmetic per pixel and nothing else.
//!
//! Two choices keep the arithmetic close to the original without using its exact expressions:
//!
//! * Angles are not converted to radians and back. Steps 1 and 6 work on unit vectors, built with
//!   products of the sines and cosines that appear in the spherical rotation of the WCS. This
//!   avoids `asin` and `acos` near the poles of the native system, where they lose digits (wcslib
//!   avoids them for the same reason).
//! * The law of cosines in step 2 is written without the cancellation of the original (see
//!   `source_pixel`). The original's rounding error there is a few millimetres of distance along
//!   the line of sight, which is about 1e-9 pixel at the disc centre for a large rotation and
//!   grows towards the limb, so the difference between the two is the original's noise.
//!
//! The kernel does not reproduce wcslib bit for bit and nothing here claims it does. The tests
//! compare against sunpy directly and report the largest differences.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Pixels per rayon task. Each pixel costs a few hundred cycles, so this keeps a task around a
/// few tens of microseconds.
const CHUNK: usize = 2048;

/// Arrays shorter than this run on the calling thread.
const PARALLEL_MIN: usize = 4096;

/// Length of the WCS parameter array, see [`Wcs::from_slice`].
pub const WCS_LEN: usize = 10;

/// Length of the chain parameter array, see [`Chain::from_slice`].
pub const CHAIN_LEN: usize = 24;

/// A celestial TAN projection with the reference point at the native pole, which is how wcslib
/// treats every zenithal projection. All angles are radians except the intermediate world
/// coordinates, which are in the unit of `cd` (degrees per pixel in the Python layer).
#[derive(Clone, Copy, Debug)]
pub struct Wcs {
    /// Pixel to intermediate world matrix (CDELT times PC), row major, degrees per pixel.
    pub cd: [f64; 4],
    /// Inverse of `cd`.
    pub cd_inv: [f64; 4],
    /// CRPIX with the pixel origin at 0 (that is, the FITS value minus one).
    pub crpix0: [f64; 2],
    /// Sine and cosine of the reference longitude.
    pub sin_a0: f64,
    pub cos_a0: f64,
    /// Sine and cosine of the reference latitude.
    pub sin_d0: f64,
    pub cos_d0: f64,
    /// Sine and cosine of LONPOLE.
    pub sin_lp: f64,
    pub cos_lp: f64,
}

impl Wcs {
    /// Layout: `cd11, cd12, cd21, cd22, crpix1 - 1, crpix2 - 1, crval1 (rad), crval2 (rad),
    /// sin(lonpole), cos(lonpole)`.
    pub fn from_slice(p: &[f64]) -> Result<Wcs, String> {
        if p.len() != WCS_LEN {
            return Err(format!("wcs must have {WCS_LEN} values, got {}", p.len()));
        }
        let det = p[0] * p[3] - p[1] * p[2];
        if det == 0.0 || !det.is_finite() {
            return Err("the pixel scale matrix is singular".into());
        }
        let cd_inv = [p[3] / det, -p[1] / det, -p[2] / det, p[0] / det];
        let (sin_a0, cos_a0) = p[6].sin_cos();
        let (sin_d0, cos_d0) = p[7].sin_cos();
        Ok(Wcs {
            cd: [p[0], p[1], p[2], p[3]],
            cd_inv,
            crpix0: [p[4], p[5]],
            sin_a0,
            cos_a0,
            sin_d0,
            cos_d0,
            sin_lp: p[8],
            cos_lp: p[9],
        })
    }

    /// Pixel (origin 0) to the unit vector `(cos d cos a, cos d sin a, sin d)` of the world
    /// direction, where `a` is the longitude and `d` the latitude.
    #[inline(always)]
    pub fn pixel_to_unit(&self, px: f64, py: f64) -> [f64; 3] {
        let dx = px - self.crpix0[0];
        let dy = py - self.crpix0[1];
        let x = self.cd[0] * dx + self.cd[1] * dy;
        let y = self.cd[2] * dx + self.cd[3] * dy;
        // Radius on the plane in degrees, converted to the tangent of the colatitude.
        let r = x.hypot(y);
        let rho = r.to_radians();
        let inv = 1.0 / (1.0 + rho * rho).sqrt();
        let (sin_t, cos_t) = (inv, rho * inv);
        // Native longitude phi = atan2(x, -y), and phi - lonpole.
        let (sin_p, cos_p) = if r > 0.0 { (x / r, -y / r) } else { (0.0, 1.0) };
        let sin_dp = sin_p * self.cos_lp - cos_p * self.sin_lp;
        let cos_dp = cos_p * self.cos_lp + sin_p * self.sin_lp;
        // Celestial coordinates of the native direction.
        let sin_d = sin_t * self.sin_d0 + cos_t * self.cos_d0 * cos_dp;
        let a = -cos_t * sin_dp; // cos d sin(a - a0)
        let b = sin_t * self.cos_d0 - cos_t * self.sin_d0 * cos_dp; // cos d cos(a - a0)
        [self.cos_a0 * b - self.sin_a0 * a, self.sin_a0 * b + self.cos_a0 * a, sin_d]
    }

    /// The unit vector of a world direction to a pixel (origin 0). `None` where the TAN
    /// projection is undefined (the direction is 90 degrees or more from the reference point).
    #[inline(always)]
    pub fn unit_to_pixel(&self, u: [f64; 3]) -> Option<[f64; 2]> {
        let a = u[1] * self.cos_a0 - u[0] * self.sin_a0; // cos d sin(a - a0)
        let b = u[0] * self.cos_a0 + u[1] * self.sin_a0; // cos d cos(a - a0)
        let w = u[2] * self.sin_d0 + self.cos_d0 * b; // sin of the native latitude
        // cos(theta) sin(phi - lonpole) and cos(theta) cos(phi - lonpole).
        let su = -a;
        let sv = u[2] * self.cos_d0 - self.sin_d0 * b;
        // x = cot(theta) sin(phi), y = -cot(theta) cos(phi), in radians, then degrees.
        if w.is_nan() || w <= 0.0 {
            return None;
        }
        let x = (self.sin_lp * sv + self.cos_lp * su) / w;
        let y = -(self.cos_lp * sv - self.sin_lp * su) / w;
        let (x, y) = (x.to_degrees(), y.to_degrees());
        Some([
            self.crpix0[0] + self.cd_inv[0] * x + self.cd_inv[1] * y,
            self.crpix0[1] + self.cd_inv[2] * x + self.cd_inv[3] * y,
        ])
    }
}

/// Everything else the chain needs, in metres, radians and plain matrices.
#[derive(Clone, Copy, Debug)]
pub struct Chain {
    /// Distance of the new observer from the Sun centre.
    pub r_new: f64,
    /// Solar radius used to place the pixels on the surface (the helioprojective `rsun`).
    pub rsun: f64,
    /// Rotation rate polynomial in `sin^2(lat)`, already multiplied by the time interval, in
    /// radians: `c0 + c1 sin^2 + c2 sin^4`.
    pub c: [f64; 3],
    /// Distance of the map observer from the Sun centre.
    pub r_map: f64,
    /// Heliocentric (new observer) to heliographic Stonyhurst, row major.
    pub a: [f64; 9],
    /// Heliographic Stonyhurst at the new time to heliocentric as seen by the map observer.
    pub b: [f64; 9],
}

impl Chain {
    /// Layout: `r_new, rsun, c0, c1, c2, r_map`, then `A` (9) and `B` (9) row major.
    pub fn from_slice(p: &[f64]) -> Result<Chain, String> {
        if p.len() != CHAIN_LEN {
            return Err(format!("chain must have {CHAIN_LEN} values, got {}", p.len()));
        }
        let mut a = [0.0; 9];
        let mut b = [0.0; 9];
        a.copy_from_slice(&p[6..15]);
        b.copy_from_slice(&p[15..24]);
        Ok(Chain { r_new: p[0], rsun: p[1], c: [p[2], p[3], p[4]], r_map: p[5], a, b })
    }
}

#[inline(always)]
fn mat_vec(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// The source pixel of one output pixel, or NaN in both entries.
#[inline(always)]
pub fn source_pixel(w: &Wcs, c: &Chain, px: f64, py: f64) -> [f64; 2] {
    const NAN: [f64; 2] = [f64::NAN, f64::NAN];
    // 1 and 2: direction as seen by the new observer, and the distance to the near intersection
    // with the sphere of radius `rsun`. This is the law of cosines of sunpy's
    // `Helioprojective.make_3d`, `d = R cos(a) - sqrt(rsun^2 - R^2 sin^2(a))`, with `sin^2(a)` taken
    // from the two small components of the unit vector. The original forms the discriminant as
    // `b^2 - 4c` from `R^2 cos^2(a)` and `R^2 - rsun^2`, which are both about 1e23 m^2 and cancel
    // to about 1e18 m^2, so its result carries a rounding error of a few millimetres in `d`. This
    // form has none of that cancellation.
    let u = w.pixel_to_unit(px, py);
    let disc = c.rsun * c.rsun - c.r_new * c.r_new * (u[1] * u[1] + u[2] * u[2]);
    let d = c.r_new * u[0] - disc.sqrt();
    if d.is_nan() {
        return NAN;
    }
    // Helioprojective cartesian is (d u0, d u1, d u2). Heliocentric is (y, z, R - x).
    let hcc = [d * u[1], d * u[2], c.r_new - d * u[0]];
    // 3: heliographic Stonyhurst at the new observer time.
    let h = mat_vec(&c.a, hcc);
    // 4: differential rotation as a function of latitude, then lon - rotation about the z axis.
    let r = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    let s = h[2] / r;
    let s2 = s * s;
    let rot = c.c[0] + c.c[1] * s2 + c.c[2] * (s2 * s2);
    let (sr, cr) = rot.sin_cos();
    let hr = [h[0] * cr + h[1] * sr, h[1] * cr - h[0] * sr, h[2]];
    // 5: heliocentric as seen by the map observer. Behind the Sun is NaN.
    let m = mat_vec(&c.b, hr);
    if m[2] < 0.0 {
        return NAN;
    }
    // 6: helioprojective cartesian is (R - z, x, y), normalised, then the WCS.
    let v = [c.r_map - m[2], m[0], m[1]];
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    match w.unit_to_pixel([v[0] / n, v[1] / n, v[2] / n]) {
        Some(p) => p,
        None => NAN,
    }
}

/// Fill `out` (pairs of source pixel coordinates) from `xy` (pairs of output pixel coordinates).
pub fn warp_sun_coordinates(xy: &[f64], out: &mut [f64], w: &Wcs, c: &Chain, parallel: bool) {
    debug_assert_eq!(xy.len(), out.len());
    let run = |xs: &[f64], os: &mut [f64]| {
        for (p, o) in xs.chunks_exact(2).zip(os.chunks_exact_mut(2)) {
            let q = source_pixel(w, c, p[0], p[1]);
            o[0] = q[0];
            o[1] = q[1];
        }
    };
    if parallel && xy.len() / 2 >= PARALLEL_MIN {
        xy.par_chunks(2 * CHUNK).zip(out.par_chunks_mut(2 * CHUNK)).for_each(|(xs, os)| run(xs, os));
    } else {
        run(xy, out);
    }
}

/// Entry point for Python. `xy` is an `(N, 2)` float64 array of output pixel coordinates
/// (x first, origin 0), `wcs` and `chain` follow the layouts of [`Wcs::from_slice`] and
/// [`Chain::from_slice`]. Returns an `(N, 2)` array of source pixel coordinates with NaN rows for
/// pixels that the original maps to NaN. The GIL is released during the loop.
#[pyfunction]
#[pyo3(signature = (xy, wcs, chain, parallel=true))]
fn warp_sun_coordinates_f64<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    wcs: Vec<f64>,
    chain: Vec<f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if xy.shape()[1] != 2 {
        return Err(PyValueError::new_err("xy must have shape (N, 2)"));
    }
    let w = Wcs::from_slice(&wcs).map_err(PyValueError::new_err)?;
    let c = Chain::from_slice(&chain).map_err(PyValueError::new_err)?;
    let n = xy.shape()[0];
    let input = xy.as_array();
    // A non contiguous input costs one copy.
    let owned;
    let xs: &[f64] = match input.as_slice() {
        Some(s) => s,
        None => {
            owned = input.iter().copied().collect::<Vec<f64>>();
            &owned
        }
    };
    let mut out = vec![0.0f64; 2 * n];
    py.detach(|| warp_sun_coordinates(xs, &mut out, &w, &c, parallel));
    let arr = numpy::ndarray::Array2::from_shape_vec((n, 2), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(warp_sun_coordinates_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wcs(cdelt: f64, rot: f64, crval: [f64; 2]) -> Wcs {
        let (s, c) = rot.sin_cos();
        let p = [
            cdelt * c, cdelt * s, -cdelt * s, cdelt * c, 511.5, 511.5,
            crval[0].to_radians(), crval[1].to_radians(), 0.0, -1.0,
        ];
        Wcs::from_slice(&p).unwrap()
    }

    fn lon_lat(u: [f64; 3]) -> (f64, f64) {
        (u[1].atan2(u[0]), u[2].atan2(u[0].hypot(u[1])))
    }

    #[test]
    fn reference_pixel_maps_to_the_reference_direction() {
        let w = wcs(1.0 / 3600.0, 0.0, [0.0009, 0.0004]);
        let u = w.pixel_to_unit(511.5, 511.5);
        let (lon, lat) = lon_lat(u);
        assert!((lon.to_degrees() - 0.0009).abs() < 1e-14);
        assert!((lat.to_degrees() - 0.0004).abs() < 1e-14);
    }

    #[test]
    fn pixel_round_trip() {
        for (rot, crval) in [(0.0, [0.0, 0.0]), (0.3, [0.0009, 0.0004]), (-1.0, [0.5, -0.2])] {
            let w = wcs(0.6 / 3600.0, rot, crval);
            for &(x, y) in &[(0.0, 0.0), (1023.0, 0.0), (10.5, 900.25), (511.5, 511.5), (777.0, 3.0)] {
                let u = w.pixel_to_unit(x, y);
                let q = w.unit_to_pixel(u).unwrap();
                assert!((q[0] - x).abs() < 1e-9 && (q[1] - y).abs() < 1e-9, "{x} {y} -> {q:?}");
            }
        }
    }

    #[test]
    fn tan_matches_the_closed_form_on_the_equator() {
        // With the reference point at (0, 0) and lonpole 180 degrees, x is -tan of the longitude
        // offset... checked against the textbook TAN formula x = tan(lon) / cos... on the axes.
        let w = wcs(1.0 / 3600.0, 0.0, [0.0, 0.0]);
        // 100 arcsec along +x of the plane is a direction at longitude atan(100 arcsec in rad).
        let u = w.pixel_to_unit(511.5 + 100.0, 511.5);
        let (lon, lat) = lon_lat(u);
        let expected = (100.0f64 / 3600.0).to_radians().atan();
        assert!((lon - expected).abs() < 1e-15, "{lon} {expected}");
        assert!(lat.abs() < 1e-15);
        let u = w.pixel_to_unit(511.5, 511.5 + 100.0);
        let (lon, lat) = lon_lat(u);
        assert!(lon.abs() < 1e-15);
        assert!((lat - expected).abs() < 1e-15);
    }

    #[test]
    fn undefined_beyond_ninety_degrees() {
        let w = wcs(1.0 / 3600.0, 0.0, [0.0, 0.0]);
        assert!(w.unit_to_pixel([-1.0, 0.0, 0.0]).is_none());
        assert!(w.unit_to_pixel([0.0, 1.0, 0.0]).is_none());
        assert!(w.unit_to_pixel([1.0, 0.0, 0.0]).is_some());
    }

    fn identity_chain(r: f64, rsun: f64) -> Chain {
        // A takes (heliocentric x, y, z) with z along the line of sight to heliographic with x
        // along the line of sight. B undoes it. No rotation, so the chain is the identity.
        let a = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        Chain { r_new: r, rsun, c: [0.0; 3], r_map: r, a, b }
    }

    #[test]
    fn zero_rotation_and_same_observer_is_the_identity() {
        let w = wcs(0.6 / 3600.0, 0.1, [0.0, 0.0]);
        let c = identity_chain(1.5e11, 6.957e8);
        for &(x, y) in &[(511.5, 511.5), (400.0, 600.0), (300.25, 250.5), (700.0, 520.0)] {
            let q = source_pixel(&w, &c, x, y);
            assert!((q[0] - x).abs() < 1e-8 && (q[1] - y).abs() < 1e-8, "{x} {y} -> {q:?}");
        }
    }

    #[test]
    fn off_disk_pixels_are_nan() {
        let w = wcs(0.6 / 3600.0, 0.0, [0.0, 0.0]);
        let c = identity_chain(1.5e11, 6.957e8);
        // 1100 arcsec from the centre is beyond the 960 arcsec limb.
        let q = source_pixel(&w, &c, 511.5 + 1100.0 / 0.6, 511.5);
        assert!(q[0].is_nan() && q[1].is_nan());
    }

    #[test]
    fn a_small_rotation_moves_the_centre_by_the_projected_arc() {
        // The identity chain puts heliographic longitude 0 at the centre of the disc. Rotating by
        // 10 degrees takes the source for the centre pixel to longitude -10 degrees, which sits at
        // x = -R sin(10 deg) from the axis, seen from a distance r - R cos(10 deg) along it.
        let w = wcs(0.6 / 3600.0, 0.0, [0.0, 0.0]);
        let (r, rsun) = (1.5e11, 6.957e8);
        let mut c = identity_chain(r, rsun);
        c.c = [10.0f64.to_radians(), 0.0, 0.0];
        let q = source_pixel(&w, &c, 511.5, 511.5);
        let rot = 10.0f64.to_radians();
        let tan_angle = rsun * rot.sin() / (r - rsun * rot.cos());
        let expect_x = 511.5 - tan_angle.atan().tan().to_degrees() / (0.6 / 3600.0);
        assert!((q[1] - 511.5).abs() < 1e-6);
        assert!((q[0] - expect_x).abs() < 1e-6, "{} {}", q[0], expect_x);
    }

    #[test]
    fn nan_input_gives_nan() {
        let w = wcs(0.6 / 3600.0, 0.0, [0.0, 0.0]);
        let c = identity_chain(1.5e11, 6.957e8);
        let q = source_pixel(&w, &c, f64::NAN, 3.0);
        assert!(q[0].is_nan() && q[1].is_nan());
    }

    #[test]
    fn parallel_and_serial_agree_bit_for_bit() {
        let w = wcs(0.6 / 3600.0, 0.05, [0.0009, 0.0004]);
        let mut c = identity_chain(1.5e11, 6.957e8);
        c.c = [0.1, -0.02, -0.01];
        let n = 3 * CHUNK + 17;
        let mut xy = Vec::with_capacity(2 * n);
        for i in 0..n {
            xy.push((i % 1024) as f64);
            xy.push((i / 1024) as f64 + 400.0);
        }
        let mut a = vec![0.0; 2 * n];
        let mut b = vec![0.0; 2 * n];
        warp_sun_coordinates(&xy, &mut a, &w, &c, false);
        warp_sun_coordinates(&xy, &mut b, &w, &c, true);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        }
    }

    #[test]
    fn wrong_lengths_are_rejected() {
        assert!(Wcs::from_slice(&[0.0; 9]).is_err());
        assert!(Chain::from_slice(&[0.0; 23]).is_err());
        assert!(Wcs::from_slice(&[0.0; WCS_LEN]).is_err());
    }
}
