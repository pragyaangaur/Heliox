//! Projection between heliocentric and helioprojective coordinates, and the
//! geometry of great arcs and limbs on the solar surface.

use std::f64::consts::TAU;

use super::linalg::{cross, dot, norm, normalise, Vec3};

/// Helioprojective longitude, latitude and distance of a point given in
/// heliocentric Cartesian coordinates, seen by an observer at `distance_to_sun`.
///
/// The arrays are paired element by element. The return value holds the
/// longitude `tx` and latitude `ty` in radians, and the distance from the
/// observer, in the same unit as the inputs.
pub fn heliocentric_to_helioprojective(
    x: f64,
    y: f64,
    z: f64,
    distance_to_sun: f64,
) -> (f64, f64, f64) {
    let along_line_of_sight = distance_to_sun - z;
    let distance = (x * x + y * y + along_line_of_sight * along_line_of_sight).sqrt();
    let tx = x.atan2(along_line_of_sight);
    let ty = (y / distance).asin();
    (tx, ty, distance)
}

/// The heliocentric Cartesian `(x, y, z)` of a point given by helioprojective
/// longitude `tx` and latitude `ty` in radians, at `distance` from the observer.
pub fn helioprojective_to_heliocentric(
    tx: f64,
    ty: f64,
    distance: f64,
    distance_to_sun: f64,
) -> (f64, f64, f64) {
    let x = distance * ty.cos() * tx.sin();
    let y = distance * ty.sin();
    let z = distance_to_sun - distance * ty.cos() * tx.cos();
    (x, y, z)
}

/// The shape of a great arc between two points on a sphere.
#[derive(Debug, PartialEq)]
pub struct ArcBasis {
    /// The length of the vector from the centre of the sphere to the start.
    pub radius: f64,
    /// The unit vector towards the start of the arc.
    pub start_hat: Vec3,
    /// The unit vector in the plane of the arc, a quarter turn from `start_hat`.
    pub quarter_hat: Vec3,
    /// The angle the arc subtends at the centre, in radians, in [0, 2 pi).
    pub angle: f64,
}

/// The basis of the great arc from `v1` to `v2`, both given relative to the
/// centre of the sphere.
pub fn great_arc_basis(v1: Vec3, v2: Vec3) -> ArcBasis {
    let radius = norm(v1);
    let v1_hat = normalise(v1);
    let v2_hat = normalise(v2);

    let along = dot(v2_hat, v1_hat);
    let perpendicular = [
        v2_hat[0] - along * v1_hat[0],
        v2_hat[1] - along * v1_hat[1],
        v2_hat[2] - along * v1_hat[2],
    ];
    let mut quarter = perpendicular;
    let mut length = norm(quarter);
    if length == 0.0 {
        // The two ends are parallel or antiparallel, so any perpendicular will
        // do. Pick one deterministically.
        quarter = cross(v1_hat, [0.0, 0.0, 1.0]);
        length = norm(quarter);
        if length == 0.0 {
            quarter = cross(v1_hat, [0.0, 1.0, 0.0]);
            length = norm(quarter);
        }
    }
    let quarter_hat = quarter.map(|component| component / length);

    let mut angle = dot(v2_hat, quarter_hat).atan2(along);
    if angle < 0.0 {
        angle += TAU;
    }
    ArcBasis {
        radius,
        start_hat: v1_hat,
        quarter_hat,
        angle,
    }
}

/// The point at each angle `angles[i]` along an arc of the given radius,
/// centred on `center`, returned flat as `x0, y0, z0, x1, ...`.
pub fn great_arc_points(basis: &ArcBasis, center: Vec3, angles: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(angles.len() * 3);
    for &angle in angles {
        let (sin, cos) = angle.sin_cos();
        for axis in 0..3 {
            let offset = basis.start_hat[axis] * cos + basis.quarter_hat[axis] * sin;
            out.push(basis.radius * offset + center[axis]);
        }
    }
    out
}

/// The `(x, y)` points of the limb circle at each angle, with the constant
/// `z` of the circle's plane. The circle has radius `rsun * sqrt(1 - ratio^2)`
/// and sits at `z = rsun * ratio`, where `ratio` is the solar radius over the
/// observer's distance.
pub fn limb_circle(rsun: f64, ratio: f64, angles: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let radius = rsun * (1.0 - ratio * ratio).sqrt();
    let xs = angles.iter().map(|angle| radius * angle.cos()).collect();
    let ys = angles.iter().map(|angle| radius * angle.sin()).collect();
    (xs, ys, rsun * ratio)
}

/// The arc length on a sphere of radius `distance` subtending each angle, in
/// the same unit as `distance`.
pub fn angle_to_distance(angle: f64, distance: f64) -> f64 {
    angle.tan() * distance
}

/// The angle subtended at an observer by a length `size` at `distance`.
pub fn distance_to_angle(size: f64, distance: f64) -> f64 {
    (size / distance).atan()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_centre_projects_to_zero_angles() {
        let (tx, ty, d) = heliocentric_to_helioprojective(0.0, 0.0, 0.0, 1.5);
        assert_eq!((tx, ty), (0.0, 0.0));
        assert_eq!(d, 1.5);
    }

    #[test]
    fn projection_round_trips() {
        let (tx, ty, d) = heliocentric_to_helioprojective(0.2, -0.1, 0.3, 2.0);
        let (x, y, z) = helioprojective_to_heliocentric(tx, ty, d, 2.0);
        assert!((x - 0.2).abs() < 1e-15);
        assert!((y + 0.1).abs() < 1e-15);
        assert!((z - 0.3).abs() < 1e-15);
    }

    #[test]
    fn quarter_turn_arc_has_a_right_angle() {
        let basis = great_arc_basis([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((basis.angle - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((basis.radius - 1.0).abs() < 1e-15);
    }

    #[test]
    fn arc_end_points_lie_on_the_arc() {
        let basis = great_arc_basis([0.0, 2.0, 0.0], [0.0, 0.0, 2.0]);
        let end = great_arc_points(&basis, [0.0; 3], &[basis.angle]);
        assert!(end[0].abs() < 1e-15);
        assert!((end[1]).abs() < 1e-15);
        assert!((end[2] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn limb_is_at_the_solar_radius() {
        let ratio = 0.01;
        let (xs, ys, z) = limb_circle(1.0, ratio, &[0.0]);
        let r = (xs[0] * xs[0] + ys[0] * ys[0] + z * z).sqrt();
        assert!((r - 1.0).abs() < 1e-14);
    }

    #[test]
    fn equivalency_round_trips() {
        let d = distance_to_angle(angle_to_distance(0.001, 1.0e8), 1.0e8);
        assert!((d - 0.001).abs() < 1e-15);
    }
}
