//! GOES flare classes, as used by `heliox.timeseries.sources.goes.flare_class`.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// The lower bound of each class in watts per square metre, in increasing order.
const CLASSES: [(char, f64); 5] = [
    ('A', 1e-8),
    ('B', 1e-7),
    ('C', 1e-6),
    ('M', 1e-5),
    ('X', 1e-4),
];

/// The label for a flux that is not positive or not finite, and for any flux below the A class.
const QUIET: &str = "A0.0";

/// Returns the GOES class of one flux, such as `C5.4`.
///
/// Each letter is a factor of ten, and the number is the flux within that decade. Fluxes above
/// the X threshold stay in the X class, so an X15 is reported as `X15.0`.
pub fn label(flux: f64) -> String {
    if !flux.is_finite() || flux <= 0.0 {
        return QUIET.to_string();
    }
    match CLASSES.iter().rev().find(|(_, lower)| flux >= *lower) {
        Some((letter, lower)) => format!("{letter}{:.1}", flux / lower),
        None => QUIET.to_string(),
    }
}

/// Returns the GOES class of every flux in the array.
#[pyfunction]
pub fn flare_labels(flux: PyReadonlyArray1<'_, f64>) -> Vec<String> {
    flux.as_array().iter().map(|&value| label(value)).collect()
}

#[cfg(test)]
mod tests {
    use super::label;

    #[test]
    fn classes_follow_the_docstring_examples() {
        assert_eq!(label(5.4e-6), "C5.4");
        assert_eq!(label(2.3e-4), "X2.3");
        assert_eq!(label(1.5e-5), "M1.5");
    }

    #[test]
    fn x_class_does_not_roll_over() {
        assert_eq!(label(1.5e-3), "X15.0");
    }

    #[test]
    fn lower_bound_belongs_to_its_own_class() {
        assert_eq!(label(1e-8), "A1.0");
        assert_eq!(label(1e-7), "B1.0");
    }

    #[test]
    fn quiet_values_are_a_zero_class() {
        for value in [0.0, -1.0, 1e-9, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(label(value), "A0.0");
        }
    }
}
