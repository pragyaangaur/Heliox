//! Empirical laws for the solar surface: differential rotation, limb darkening
//! and the sunspot number to radio flux relation.

use super::ephemeris::DEG;

/// The synodic correction to the sidereal rotation rate, in degrees per day.
/// The Earth moves along its orbit, so a feature has to turn this much further
/// to face the Earth again.
const SYNODIC_CORRECTION: f64 = 0.9856;

/// The angle in degrees through which a feature turns in `days` days, at a
/// latitude in degrees, under the rotation law `a + b sin^2 + c sin^4`.
///
/// `a`, `b` and `c` are in degrees per day. Set `synodic` to subtract the
/// Earth's orbital motion from the rate.
pub fn rotation_angle(latitude: f64, days: f64, a: f64, b: f64, c: f64, synodic: bool) -> f64 {
    let s = (latitude * DEG).sin();
    let s2 = s * s;
    let mut rate = a + b * s2 + c * (s2 * s2);
    if synodic {
        rate -= SYNODIC_CORRECTION;
    }
    rate * days
}

/// The quadratic limb darkening intensity relative to disc centre at radial
/// position `r` (a fraction of the solar radius), for coefficients `u1` and
/// `u2`. Points off the disc give zero, and NaN gives zero.
pub fn limb_darkening(r: f64, u1: f64, u2: f64) -> f64 {
    if r.is_nan() || r > 1.0 {
        return 0.0;
    }
    let mu = (1.0 - r * r).sqrt();
    let intensity = 1.0 - u1 * (1.0 - mu) - u2 * ((1.0 - mu) * (1.0 - mu));
    if intensity < 0.0 {
        0.0
    } else {
        intensity
    }
}

/// Wavelengths in angstrom at which the limb darkening coefficients are tabulated.
const LIMB_GRID: [f64; 8] = [
    3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 10000.0, 20000.0,
];
/// The linear coefficient `u1` at each wavelength in `LIMB_GRID`.
const LIMB_U1: [f64; 8] = [0.88, 0.93, 0.83, 0.72, 0.65, 0.58, 0.50, 0.35];
/// The quadratic coefficient `u2` at each wavelength in `LIMB_GRID`.
const LIMB_U2: [f64; 8] = [-0.23, -0.23, -0.17, -0.10, -0.08, -0.06, -0.05, -0.03];

/// The quadratic limb darkening coefficients `(u1, u2)` at a wavelength in
/// angstrom, interpolated linearly from the tabulated values and held constant
/// beyond the ends of the table.
///
/// The values follow Pierce & Slaughter (1977) and Neckel & Labs (1994).
pub fn limb_darkening_coefficients(angstrom: f64) -> (f64, f64) {
    (
        interp(angstrom, &LIMB_GRID, &LIMB_U1),
        interp(angstrom, &LIMB_GRID, &LIMB_U2),
    )
}

/// Linear interpolation with the behaviour of `numpy.interp`: values outside
/// the grid take the nearest end value, and NaN stays NaN.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let last = xs.len() - 1;
    if x.is_nan() {
        f64::NAN
    } else if x < xs[0] {
        ys[0]
    } else if x >= xs[last] {
        ys[last]
    } else {
        let j = xs.partition_point(|&grid| grid <= x) - 1;
        let slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
        slope * (x - xs[j]) + ys[j]
    }
}

/// The 10.7 cm radio flux in solar flux units from the international sunspot
/// number, using the quadratic relation of Holland & Vaughan (1984).
pub fn sunspot_flux(sunspot_number: f64) -> f64 {
    63.7 + 0.728 * sunspot_number + 0.00089 * (sunspot_number * sunspot_number)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_equator_rotates_at_the_sidereal_rate() {
        let angle = rotation_angle(0.0, 1.0, 14.1844, 0.0, 0.0, false);
        assert!((angle - 14.1844).abs() < 1e-12);
    }

    #[test]
    fn the_synodic_rate_is_slower_by_the_orbital_motion() {
        let sidereal = rotation_angle(30.0, 2.0, 14.44, -3.0, 0.0, false);
        let synodic = rotation_angle(30.0, 2.0, 14.44, -3.0, 0.0, true);
        assert!((sidereal - synodic - 2.0 * SYNODIC_CORRECTION).abs() < 1e-12);
    }

    #[test]
    fn limb_darkening_is_unity_at_centre_and_zero_off_disc() {
        assert_eq!(limb_darkening(0.0, 0.72, -0.10), 1.0);
        assert_eq!(limb_darkening(1.5, 0.72, -0.10), 0.0);
        assert_eq!(limb_darkening(f64::NAN, 0.72, -0.10), 0.0);
    }

    #[test]
    fn coefficients_are_interpolated_and_clamped() {
        let (u1, u2) = limb_darkening_coefficients(6000.0);
        assert_eq!((u1, u2), (0.72, -0.10));
        let (low, _) = limb_darkening_coefficients(1000.0);
        assert_eq!(low, 0.88);
        let (mid, _) = limb_darkening_coefficients(5500.0);
        assert!((mid - 0.775).abs() < 1e-12);
    }

    #[test]
    fn quiet_sun_flux_is_the_intercept() {
        assert_eq!(sunspot_flux(0.0), 63.7);
    }
}
