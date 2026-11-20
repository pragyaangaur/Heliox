//! Arithmetic on times held as two doubles, following astropy's `time/utils.py`.
//!
//! A time is a pair `(jd1, jd2)` of Julian dates. The routines here add, scale
//! and divide such pairs with exact error-free transformations (Shewchuk, 1997),
//! and then split the result into an integer day and a remainder in `[-0.5, 0.5]`.
//! Every operation is written in the same order as the Python original, so the
//! results agree bit for bit.

/// Adds `a` and `b` exactly. Returns the rounded sum and the rounding error.
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let eb = x - a;
    let ea = x - eb;
    let eb = b - eb;
    let ea = a - ea;
    (x, ea + eb)
}

/// Splits `a` into two halves whose products with other halves are exact.
pub fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0 * a;
    let abig = c - a;
    let ah = c - abig;
    let al = a - ah;
    (ah, al)
}

/// Multiplies `a` and `b` exactly. Returns the rounded product and its error.
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let y1 = ah * bh;
    let mut y = x - y1;
    let y2 = al * bh;
    y -= y2;
    let y3 = ah * bl;
    y -= y3;
    let y4 = al * bl;
    y = y4 - y;
    (x, y)
}

/// The sign of `x` as `numpy.sign` returns it: `-1`, `0` or `1`, and `NaN` for `NaN`.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Returns the sum `val1 + val2` as an integer day and a remainder.
///
/// The sum can be multiplied by `factor` or divided by `divisor` first. At most
/// one of them is `Some`. The remainder lies in `[-0.5, 0.5]`, and a tie at
/// exactly half a day rounds to the even day, as `numpy.round` does.
pub fn day_frac(val1: f64, val2: f64, factor: Option<f64>, divisor: Option<f64>) -> (f64, f64) {
    let (mut sum12, mut err12) = two_sum(val1, val2);

    if let Some(factor) = factor {
        let (product, carry) = two_product(sum12, factor);
        let carry = carry + err12 * factor;
        (sum12, err12) = two_sum(product, carry);
    }

    if let Some(divisor) = divisor {
        let q1 = sum12 / divisor;
        let (p1, p2) = two_product(q1, divisor);
        let (d1, d2) = two_sum(sum12, -p1);
        let d2 = d2 + err12;
        let d2 = d2 - p2;
        let q2 = (d1 + d2) / divisor;
        (sum12, err12) = two_sum(q1, q2);
    }

    let day = sum12.round_ties_even();
    let (frac, check) = two_sum(sum12 - day, err12);
    // When the remainder is exactly half a day, the check term decides which
    // way the rounding should have gone, so use it to correct the day.
    let excess = if frac * sign(check) != 0.5 {
        frac.round_ties_even()
    } else {
        (frac + 2.0 * check).round_ties_even()
    };
    let day = day + excess;
    let frac = (sum12 - day) + err12;
    (day, frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sum_recovers_the_rounding_error() {
        assert_eq!(two_sum(1e16, 1.0), (1e16, 1.0));
        assert_eq!(two_sum(0.5, 0.25), (0.75, 0.0));
    }

    #[test]
    fn two_product_recovers_the_rounding_error() {
        let (x, y) = two_product(1.0 + 2f64.powi(-30), 1.0 + 2f64.powi(-30));
        assert_eq!(x, (1.0 + 2f64.powi(-30)) * (1.0 + 2f64.powi(-30)));
        assert_eq!(y, 2f64.powi(-60));
    }

    #[test]
    fn day_frac_splits_a_sum_into_day_and_fraction() {
        assert_eq!(day_frac(0.5, 0.5, None, None), (1.0, 0.0));
        assert_eq!(day_frac(2_458_000.0, 0.25, None, None), (2_458_000.0, 0.25));
        assert_eq!(
            day_frac(2_458_000.0, 0.75, None, None),
            (2_458_001.0, -0.25)
        );
    }

    #[test]
    fn day_frac_rounds_a_half_day_to_the_even_day() {
        assert_eq!(day_frac(2.5, 0.0, None, None), (2.0, 0.5));
        assert_eq!(day_frac(3.5, 0.0, None, None), (4.0, -0.5));
    }

    #[test]
    fn day_frac_applies_a_factor_and_a_divisor() {
        assert_eq!(day_frac(1.5, 0.0, Some(4.0), None), (6.0, 0.0));
        assert_eq!(day_frac(86400.0, 0.0, None, Some(86400.0)), (1.0, 0.0));
        assert_eq!(day_frac(1.0, 0.0, None, Some(2.0)), (0.0, 0.5));
    }
}
