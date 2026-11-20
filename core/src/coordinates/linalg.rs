//! Fixed-size vector and matrix helpers.
//!
//! Each operation adds its terms in the same order that NumPy uses for the
//! three-element arrays this code replaces, so that the results agree to the
//! last bit wherever NumPy's own summation is sequential.

/// A three-component vector.
pub type Vec3 = [f64; 3];

/// A 3 by 3 matrix, stored row by row.
pub type Mat3 = [[f64; 3]; 3];

/// The dot product of two vectors.
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The cross product `a x b`.
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The Euclidean length of a vector.
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// The vector scaled to unit length.
pub fn normalise(a: Vec3) -> Vec3 {
    let length = norm(a);
    a.map(|component| component / length)
}

/// The part of `v` perpendicular to the unit vector `axis`, scaled to unit length.
pub fn unit_perpendicular(v: Vec3, axis: Vec3) -> Vec3 {
    let along = dot(v, axis);
    normalise([
        v[0] - along * axis[0],
        v[1] - along * axis[1],
        v[2] - along * axis[2],
    ])
}

/// The product of two 3 by 3 matrices, `a @ b`.
pub fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Python's floating point remainder, whose result takes the sign of `b`.
pub fn python_mod(a: f64, b: f64) -> f64 {
    let remainder = a % b;
    if remainder == 0.0 {
        0.0_f64.copysign(b)
    } else if (b < 0.0) != (remainder < 0.0) {
        remainder + b
    } else {
        remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_follows_the_right_hand_rule() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn normalise_gives_unit_length() {
        assert!((norm(normalise([3.0, 4.0, 12.0])) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn unit_perpendicular_removes_the_axis_component() {
        let p = unit_perpendicular([1.0, 1.0, 1.0], [0.0, 0.0, 1.0]);
        assert!(p[2].abs() < 1e-15);
        assert!((norm(p) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn python_mod_takes_the_sign_of_the_divisor() {
        assert_eq!(python_mod(-1.0, 360.0), 359.0);
        assert_eq!(python_mod(721.5, 360.0), 1.5);
        assert_eq!(python_mod(0.0, 360.0), 0.0);
    }

    #[test]
    fn matmul_with_identity_is_unchanged() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert_eq!(matmul(&identity, &m), m);
        assert_eq!(matmul(&m, &identity), m);
    }
}
