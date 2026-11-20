//! The arithmetic behind `heliox.time.TimeRange`, on times held as two doubles.
//!
//! The Python side keeps the astropy `Time` objects and converts them between
//! scales. Each function here does one step of astropy's arithmetic on the raw
//! `(jd1, jd2)` pairs. The steps follow the order in which astropy performs
//! them, so a range built from these results matches the one built by astropy
//! bit for bit.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

use super::twodouble::day_frac;

/// Adds two time pairs, as `Time.__add__` and `TimeDelta._add_sub` do.
#[pyfunction]
pub fn jd_add(a1: f64, a2: f64, b1: f64, b2: f64) -> (f64, f64) {
    day_frac(a1 + b1, a2 + b2, None, None)
}

/// Subtracts a time pair from another, as `Time.__sub__` does for a `TimeDelta`.
#[pyfunction]
pub fn jd_sub(a1: f64, a2: f64, b1: f64, b2: f64) -> (f64, f64) {
    day_frac(a1 - b1, a2 - b2, None, None)
}

/// Multiplies a time pair by a number, as `TimeDelta.__mul__` does.
#[pyfunction]
pub fn jd_scale(a1: f64, a2: f64, factor: f64) -> (f64, f64) {
    day_frac(a1, a2, Some(factor), None)
}

/// Divides a time pair by a number, as `TimeDelta.__truediv__` does.
#[pyfunction]
pub fn jd_divide(a1: f64, a2: f64, divisor: f64) -> (f64, f64) {
    day_frac(a1, a2, None, Some(divisor))
}

/// Subtracts one `Time` from another in a common scale, as `Time.__sub__` does.
///
/// The first pair is first normalised by dividing by one, as the `TimeDelta`
/// constructor does, and then the second is subtracted and the result normalised.
#[pyfunction]
pub fn jd_difference(a1: f64, a2: f64, b1: f64, b2: f64) -> (f64, f64) {
    let (n1, n2) = day_frac(a1, a2, None, Some(1.0));
    day_frac(n1 - b1, n2 - b2, None, None)
}

/// The signed difference of two time pairs, as `Time._time_comparison` forms it.
///
/// The comparison is `compare(a, b) < 0` and so on, so the sign of the result
/// is what matters.
#[pyfunction]
pub fn jd_compare(a1: f64, a2: f64, b1: f64, b2: f64) -> f64 {
    (a1 - b1) + (a2 - b2)
}

/// A time pair expressed in a unit: its sum times the number of units in a day.
#[pyfunction]
pub fn jd_value(a1: f64, a2: f64, factor: f64) -> f64 {
    (a1 + a2) * factor
}

/// The number of windows of length `cadence` that a duration of `length` holds.
///
/// This is `int(floor(length / cadence)) + 1`. Both arguments are time pairs.
#[pyfunction]
pub fn window_count(l1: f64, l2: f64, c1: f64, c2: f64) -> PyResult<i64> {
    count_from_floor((l1 + l2) / (c1 + c2)).map(|n| n + 1)
}

/// The number of calendar days a range touches, given the MJDs of its ends.
///
/// This is `int(floor(end - floor(start))) + 1`.
#[pyfunction]
pub fn day_count(end_mjd: f64, start_mjd: f64) -> PyResult<i64> {
    count_from_floor(end_mjd - start_mjd.floor()).map(|n| n + 1)
}

/// Converts `floor(value)` to an integer, raising the errors `int` raises for
/// values that have no integer form.
fn count_from_floor(value: f64) -> PyResult<i64> {
    let floored = value.floor();
    if floored.is_nan() {
        Err(PyValueError::new_err("cannot convert float NaN to integer"))
    } else if floored.is_infinite() {
        Err(PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ))
    } else if floored.abs() >= 9.2e18 {
        Err(PyOverflowError::new_err("count is too large"))
    } else {
        Ok(floored as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_count_includes_the_window_at_the_end() {
        // A 60 minute range with 20 minute cadence holds windows at 0, 20, 40 and 60.
        let minute = 1.0 / 1440.0;
        assert_eq!(
            window_count(60.0 * minute, 0.0, 20.0 * minute, 0.0).unwrap(),
            4
        );
    }

    #[test]
    fn day_count_counts_each_calendar_day_touched() {
        assert_eq!(day_count(58_052.9, 58_051.1).unwrap(), 2);
        assert_eq!(day_count(58_051.5, 58_051.5).unwrap(), 1);
    }

    #[test]
    fn count_from_floor_rejects_values_without_an_integer_form() {
        assert!(count_from_floor(f64::NAN).is_err());
        assert!(count_from_floor(f64::INFINITY).is_err());
        assert_eq!(count_from_floor(-0.5).unwrap(), -1);
    }
}
