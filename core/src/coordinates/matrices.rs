//! Rotation matrices relating the solar frames to the ICRS axes.
//!
//! The Sun's pole and prime meridian follow the IAU 2015 rotational elements.
//! Every matrix has its rows set to the basis vectors of the new frame, so a
//! vector in ICRS axes multiplied by the matrix gives its components in that
//! frame.

use super::linalg::{cross, python_mod, unit_perpendicular, Mat3, Vec3};

/// The right ascension of the Sun's pole in the ICRF, in degrees.
const POLE_RA_DEG: f64 = 286.13;

/// The declination of the Sun's pole in the ICRF, in degrees.
const POLE_DEC_DEG: f64 = 63.87;

/// The prime meridian angle `W` at J2000, in degrees.
const PRIME_MERIDIAN_AT_J2000_DEG: f64 = 84.176;

/// The rotation rate of the prime meridian, in degrees per day.
const ROTATION_RATE_DEG_PER_DAY: f64 = 14.1844000;

/// A unit vector in ICRS axes from a right ascension and declination in degrees.
pub fn unit_vector(ra_deg: f64, dec_deg: f64) -> Vec3 {
    let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
    [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]
}

/// The Sun's rotation axis as a unit vector in ICRS axes.
pub fn solar_pole() -> Vec3 {
    unit_vector(POLE_RA_DEG, POLE_DEC_DEG)
}

/// The ascending node of the solar equator on the ICRF equator, as a unit vector.
pub fn solar_equator_node() -> Vec3 {
    unit_vector(POLE_RA_DEG + 90.0, 0.0)
}

/// The prime meridian angle `W`, in degrees in the range [0, 360), for a
/// number of days elapsed since J2000.
pub fn prime_meridian_degrees(elapsed_days: f64) -> f64 {
    python_mod(
        PRIME_MERIDIAN_AT_J2000_DEG + ROTATION_RATE_DEG_PER_DAY * elapsed_days,
        360.0,
    )
}

/// The matrix whose rows are the basis of a frame with the given pole, and
/// whose x axis is the component of `x_axis` perpendicular to that pole.
pub fn frame_from_pole_and_x(x_axis: Vec3, pole: Vec3) -> Mat3 {
    let x_hat = unit_perpendicular(x_axis, pole);
    [x_hat, cross(pole, x_hat), pole]
}

/// The matrix taking ICRS axes into Carrington axes at the given prime
/// meridian angle in degrees.
pub fn carrington_matrix(angle_deg: f64) -> Mat3 {
    let (pole, node) = (solar_pole(), solar_equator_node());
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let ahead = cross(pole, node);
    let x_axis = [
        node[0] * cos + ahead[0] * sin,
        node[1] * cos + ahead[1] * sin,
        node[2] * cos + ahead[2] * sin,
    ];
    frame_from_pole_and_x(x_axis, pole)
}

/// The matrix taking ICRS axes into heliocentric inertial axes. The x axis is
/// the solar equator node, which does not move.
pub fn inertial_matrix() -> Mat3 {
    frame_from_pole_and_x(solar_equator_node(), solar_pole())
}

/// The matrix taking Stonyhurst heliographic axes into heliocentric axes for
/// an observer at the given longitude and latitude in radians. The z axis
/// points at the observer and the y axis at solar north.
pub fn heliocentric_matrix(lon: f64, lat: f64) -> Mat3 {
    let z_hat = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let y_hat = unit_perpendicular([0.0, 0.0, 1.0], z_hat);
    [cross(y_hat, z_hat), y_hat, z_hat]
}

#[cfg(test)]
mod tests {
    use super::super::linalg::{dot, norm};
    use super::*;

    fn assert_orthonormal(m: &Mat3) {
        for i in 0..3 {
            assert!((norm(m[i]) - 1.0).abs() < 1e-14);
            for j in 0..i {
                assert!(dot(m[i], m[j]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn solar_pole_is_a_unit_vector_near_the_ecliptic_north() {
        let pole = solar_pole();
        assert!((norm(pole) - 1.0).abs() < 1e-15);
        assert!((pole[2] - 63.87_f64.to_radians().sin()).abs() < 1e-15);
    }

    #[test]
    fn prime_meridian_at_j2000_is_the_iau_value() {
        assert!((prime_meridian_degrees(0.0) - 84.176).abs() < 1e-12);
    }

    #[test]
    fn prime_meridian_wraps_into_the_principal_range() {
        let w = prime_meridian_degrees(-1000.0);
        assert!((0.0..360.0).contains(&w));
    }

    #[test]
    fn every_matrix_is_a_rotation() {
        assert_orthonormal(&carrington_matrix(123.4));
        assert_orthonormal(&inertial_matrix());
        assert_orthonormal(&heliocentric_matrix(0.3, -0.2));
        assert_orthonormal(&frame_from_pole_and_x([1.0, 2.0, 3.0], solar_pole()));
    }

    #[test]
    fn heliocentric_z_points_at_the_observer() {
        let m = heliocentric_matrix(0.5, 0.25);
        let z = [
            0.25_f64.cos() * 0.5_f64.cos(),
            0.25_f64.cos() * 0.5_f64.sin(),
            0.25_f64.sin(),
        ];
        assert!(norm([m[2][0] - z[0], m[2][1] - z[1], m[2][2] - z[2]]) < 1e-15);
    }
}
