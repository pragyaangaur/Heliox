//! Number formatting that matches Python's `format` mini-language.
//!
//! The map summary and the default titles print numbers with `.4f`, `.1f`, `.4g` and
//! `g`. Rust has no general format that behaves like Python's, so the rules are
//! written out here.

use pyo3::prelude::*;

/// The Python spelling of a non-finite number, or `None` if the number is finite.
fn non_finite(value: f64) -> Option<String> {
    if value.is_nan() {
        Some("nan".to_string())
    } else if value.is_infinite() {
        Some(if value > 0.0 { "inf" } else { "-inf" }.to_string())
    } else {
        None
    }
}

/// Removes trailing zeros after a decimal point, and the point itself if nothing follows it.
fn trim_zeros(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

/// A number with a fixed count of decimal places, as Python writes `format(value, ".<n>f")`.
#[pyfunction]
pub fn format_fixed(value: f64, decimals: usize) -> String {
    non_finite(value).unwrap_or_else(|| format!("{value:.decimals$}"))
}

/// A number with `significant` digits in Python's general format, as `format(value, ".<n>g")`.
///
/// Trailing zeros are dropped. The number is written in exponent form when its
/// exponent is below -4 or at least `significant`, and in fixed form otherwise.
#[pyfunction]
pub fn format_general(value: f64, significant: usize) -> String {
    if let Some(text) = non_finite(value) {
        return text;
    }
    let significant = significant.max(1);
    let scientific = format!("{:.*e}", significant - 1, value);
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust always writes exponent form with an 'e'");
    let exponent: i32 = exponent
        .parse()
        .expect("Rust writes the exponent as an integer");
    if exponent < -4 || exponent >= significant as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            trim_zeros(mantissa),
            exponent.unsigned_abs()
        )
    } else {
        let decimals = (significant as i32 - 1 - exponent) as usize;
        trim_zeros(&format!("{value:.decimals$}")).to_string()
    }
}

/// Adds this file's functions to the `maps` module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(format_fixed, module)?)?;
    module.add_function(wrap_pyfunction!(format_general, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_places_the_requested_decimals() {
        assert_eq!(format_fixed(1.23456, 4), "1.2346");
        assert_eq!(format_fixed(-0.0, 2), "-0.00");
        assert_eq!(format_fixed(512.0, 1), "512.0");
    }

    #[test]
    fn fixed_spells_non_finite_values_as_python_does() {
        assert_eq!(format_fixed(f64::NAN, 3), "nan");
        assert_eq!(format_fixed(f64::INFINITY, 3), "inf");
        assert_eq!(format_fixed(f64::NEG_INFINITY, 3), "-inf");
    }

    #[test]
    fn general_uses_fixed_form_inside_the_range() {
        assert_eq!(format_general(171.0, 6), "171");
        assert_eq!(format_general(0.5, 4), "0.5");
        assert_eq!(format_general(-12.3456, 4), "-12.35");
        assert_eq!(format_general(0.0, 4), "0");
    }

    #[test]
    fn general_switches_to_exponent_form_at_the_edges() {
        assert_eq!(format_general(1e20, 4), "1e+20");
        assert_eq!(format_general(1.5e-5, 4), "1.5e-05");
        assert_eq!(format_general(0.0001, 4), "0.0001");
        assert_eq!(format_general(123456.0, 6), "123456");
        assert_eq!(format_general(1234567.0, 6), "1.23457e+06");
    }

    #[test]
    fn general_rounds_before_choosing_the_form() {
        assert_eq!(format_general(9.99999, 4), "10");
        assert_eq!(format_general(999999.5, 6), "1e+06");
    }

    #[test]
    fn general_spells_non_finite_values_as_python_does() {
        assert_eq!(format_general(f64::NAN, 4), "nan");
        assert_eq!(format_general(f64::NEG_INFINITY, 6), "-inf");
    }
}
