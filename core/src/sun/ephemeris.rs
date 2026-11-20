//! Solar ephemeris formulas from Meeus, *Astronomical Algorithms*, and the
//! ecliptic and rotation geometry built on them.
//!
//! Angles are in degrees unless a name says otherwise. Each function works on
//! one value, and the Python wrappers apply it to whole arrays.

use std::f64::consts::PI;

/// Radians per degree.
pub const DEG: f64 = PI / 180.0;
/// Degrees per radian.
pub const RAD_TO_DEG: f64 = 180.0 / PI;
/// Arcseconds per radian.
pub const RAD_TO_ARCSEC: f64 = 648_000.0 / PI;

/// The Julian date of the epoch J2000.0.
const J2000: f64 = 2_451_545.0;
/// Days in a Julian century.
const DAYS_PER_CENTURY: f64 = 36_525.0;
/// Inclination of the solar equator to the ecliptic, in degrees.
const INCLINATION: f64 = 7.25;
/// Julian date that the node longitude is measured from.
const NODE_EPOCH_JD: f64 = 2_396_758.0;
/// Longitude of the ascending node of the solar equator at the node epoch.
const NODE_LONGITUDE: f64 = 73.666667;
/// Rate of change of the node longitude, in degrees per Julian century.
const NODE_RATE: f64 = 1.3958333;
/// Meeus chapter 29 measures the rotation phase from this epoch, chosen so that
/// the disc-centre Carrington longitude is exactly zero at the start of
/// Carrington rotation 1.
const ROTATION_PHASE_EPOCH_JD: f64 = 2_398_220.0;
/// The sidereal rotation period adopted by Carrington, in days.
const SIDEREAL_PERIOD_DAYS: f64 = 25.38;

/// The remainder of `a / b` with the sign of `b`, exactly as Python and numpy
/// compute `a % b`.
pub fn python_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m == 0.0 {
        0.0_f64.copysign(b)
    } else if (m < 0.0) != (b < 0.0) {
        m + b
    } else {
        m
    }
}

/// Julian centuries of TT elapsed since J2000.0.
pub fn julian_centuries(jd: f64) -> f64 {
    (jd - J2000) / DAYS_PER_CENTURY
}

/// The Sun's mean anomaly from Meeus equation 25.3, in degrees from 0 to 360.
pub fn mean_anomaly(jd: f64) -> f64 {
    let c = julian_centuries(jd);
    python_mod(357.52911 + 35999.05029 * c - 0.0001537 * (c * c), 360.0)
}

/// The eccentricity of the Earth's orbit from Meeus equation 25.4.
pub fn eccentricity(jd: f64) -> f64 {
    let c = julian_centuries(jd);
    0.016708634 - 0.000042037 * c - 0.0000001267 * (c * c)
}

/// The Sun's equation of the centre in degrees: the true anomaly minus the mean
/// anomaly.
pub fn equation_of_centre(jd: f64) -> f64 {
    let c = julian_centuries(jd);
    let m = mean_anomaly(jd);
    (1.914602 - 0.004817 * c - 0.000014 * (c * c)) * (m * DEG).sin()
        + (0.019993 - 0.000101 * c) * ((2.0 * m) * DEG).sin()
        + 0.000289 * ((3.0 * m) * DEG).sin()
}

/// The Sun's true anomaly in degrees from 0 to 360.
pub fn true_anomaly(jd: f64) -> f64 {
    python_mod(mean_anomaly(jd) + equation_of_centre(jd), 360.0)
}

/// The apparent obliquity of the ecliptic in degrees, from the true obliquity
/// and Meeus equation 25.8.
pub fn apparent_obliquity(jd: f64, true_obliquity: f64) -> f64 {
    let omega = (125.04 - 1934.136 * julian_centuries(jd)) * DEG;
    true_obliquity + 0.00256 * omega.cos()
}

/// The ecliptic longitude of the ascending node of the solar equator, in degrees.
fn node_longitude(jd: f64) -> f64 {
    NODE_LONGITUDE + NODE_RATE * (jd - NODE_EPOCH_JD) / DAYS_PER_CENTURY
}

/// The heliographic latitude of the disc centre, B0, in degrees, from the
/// apparent longitude of the Sun in degrees and the Julian date.
pub fn b0(lambda: f64, jd: f64) -> f64 {
    let node = node_longitude(jd);
    let sine = ((lambda - node) * DEG).sin() * (INCLINATION * DEG).sin();
    sine.asin() * RAD_TO_DEG
}

/// The Carrington longitude of the disc centre, L0, in degrees from 0 to 360.
pub fn l0(lambda: f64, jd: f64) -> f64 {
    let node = node_longitude(jd);
    let theta = (jd - ROTATION_PHASE_EPOCH_JD) / SIDEREAL_PERIOD_DAYS * 360.0;
    let eta = (-((lambda - node) * DEG).sin() * (INCLINATION * DEG).cos())
        .atan2(-((lambda - node) * DEG).cos())
        * RAD_TO_DEG;
    python_mod(eta - theta, 360.0)
}

/// The position angle of the solar north pole, P, in degrees from -180 to 180,
/// from the apparent longitude, the apparent obliquity and the Julian date.
pub fn p_angle(lambda: f64, obliquity: f64, jd: f64) -> f64 {
    let node = node_longitude(jd);
    let x = (-(lambda * DEG).cos() * (obliquity * DEG).tan()).atan();
    let y = (-((lambda - node) * DEG).cos() * (INCLINATION * DEG).tan()).atan();
    python_mod((x + y) * RAD_TO_DEG + 180.0, 360.0) - 180.0
}

/// Rotates ecliptic longitude and latitude into right ascension and declination
/// about the given obliquity. Returns the right ascension in degrees from 0 to
/// 360 and the declination in degrees.
pub fn ecliptic_to_equatorial(lon: f64, lat: f64, obliquity: f64) -> (f64, f64) {
    let (lon, lat, obl) = (lon * DEG, lat * DEG, obliquity * DEG);
    let ra = (lon.sin() * obl.cos() - lat.tan() * obl.sin()).atan2(lon.cos());
    let dec = (lat.sin() * obl.cos() + lat.cos() * obl.sin() * lon.sin()).asin();
    (python_mod(ra * RAD_TO_DEG, 360.0), dec * RAD_TO_DEG)
}

/// The parallactic angle in degrees, from the hour angle, the declination and
/// the latitude of the observer, all in degrees.
pub fn parallactic_angle(hour_angle: f64, dec: f64, lat: f64) -> f64 {
    let (h, d, phi) = (hour_angle * DEG, dec * DEG, lat * DEG);
    h.sin().atan2(phi.tan() * d.cos() - d.sin() * h.cos()) * RAD_TO_DEG
}

/// The angle from local zenith to solar north in degrees from -180 to 180, from
/// the position angle P and the parallactic angle.
pub fn orientation(p: f64, parallactic: f64) -> f64 {
    python_mod((p - parallactic) + 180.0, 360.0) - 180.0
}

/// The angular radius of the solar disc in arcseconds, from the solar radius
/// and the Earth-Sun distance in the same units.
pub fn angular_radius(radius: f64, distance: f64) -> f64 {
    (radius / distance).asin() * RAD_TO_ARCSEC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_mod_takes_the_sign_of_the_divisor() {
        assert_eq!(python_mod(-1.0, 360.0), 359.0);
        assert_eq!(python_mod(360.0, 360.0), 0.0);
        assert_eq!(python_mod(7.5, -2.0), -0.5);
        assert!(python_mod(-360.0, 360.0).is_sign_positive());
    }

    #[test]
    fn julian_centuries_is_zero_at_j2000() {
        assert_eq!(julian_centuries(J2000), 0.0);
        assert!((julian_centuries(J2000 + DAYS_PER_CENTURY) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn mean_anomaly_at_j2000_matches_meeus() {
        assert!((mean_anomaly(J2000) - 357.52911).abs() < 1e-12);
    }

    #[test]
    fn orientation_wraps_into_the_principal_range() {
        assert!((orientation(170.0, -20.0) - (-170.0)).abs() < 1e-12);
        assert!((orientation(0.0, 0.0)).abs() < 1e-12);
    }

    #[test]
    fn ecliptic_at_zero_longitude_maps_to_the_equinox() {
        let (ra, dec) = ecliptic_to_equatorial(0.0, 0.0, 23.44);
        assert!(ra.abs() < 1e-12);
        assert!(dec.abs() < 1e-12);
    }

    #[test]
    fn b0_is_bounded_by_the_axial_tilt() {
        for day in 0..366 {
            let jd = 2_456_600.0 + f64::from(day);
            let lambda = 360.0 * f64::from(day) / 365.25;
            assert!(b0(lambda, jd).abs() <= INCLINATION + 1e-9);
        }
    }

    #[test]
    fn angular_radius_is_about_sixteen_arcminutes() {
        let radius = angular_radius(6.957e8 / 1.495978707e11, 1.0);
        assert!((radius / 60.0 - 15.99).abs() < 0.05);
    }
}
