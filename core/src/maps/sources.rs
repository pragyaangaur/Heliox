//! Header-derived values and display limits for the instrument map sources.
//!
//! The percentiles follow numpy's default `linear` method exactly, including the order of the
//! floating point operations, so that the display limits match the Python code they replace.

use std::cmp::Ordering;

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;

use super::{factory, sequence};

/// Whether an instrument keyword starts with an upper-case prefix, ignoring the case of the keyword.
#[pyfunction]
pub fn instrument_matches(keyword: &str, prefix: &str) -> bool {
    keyword.to_uppercase().starts_with(prefix)
}

/// The data product named by an HMI `CONTENT` keyword.
///
/// The product is one of `magnetogram`, `dopplergram`, `continuum` or `linedepth` when the
/// keyword names it, `continuum` for an intensity, and otherwise the lower-case keyword itself,
/// or `hmi` when the keyword is empty.
#[pyfunction]
pub fn hmi_measurement(content: &str) -> String {
    let content = content.to_lowercase();
    let squeezed: String = content
        .chars()
        .filter(|&character| character != ' ')
        .collect();
    for product in ["magnetogram", "dopplergram", "continuum", "linedepth"] {
        if squeezed.contains(product) {
            return product.to_string();
        }
    }
    if content.contains("intensity") {
        "continuum".to_string()
    } else if content.is_empty() {
        "hmi".to_string()
    } else {
        content
    }
}

/// The colour table name for a LASCO detector: `soholasco3` for C3 and `soholasco2` otherwise.
#[pyfunction]
pub fn lasco_colour_table(detector: &str) -> &'static str {
    if detector.to_uppercase() == "C3" {
        "soholasco3"
    } else {
        "soholasco2"
    }
}

/// The EIT channel, in angstroms, whose passband is nearest to a wavelength.
///
/// On a tie the earlier channel in the list wins, as Python's `min` does.
#[pyfunction]
pub fn nearest_eit_channel(wavelength: f64) -> i64 {
    const CHANNELS: [i64; 4] = [94, 171, 193, 304];
    let gap = |channel: i64| (channel as f64 - wavelength).abs();
    let mut nearest = CHANNELS[0];
    for &channel in &CHANNELS[1..] {
        if gap(channel) < gap(nearest) {
            nearest = channel;
        }
    }
    nearest
}

/// The display limits for an AIA image, as `(vmin, vmax)`, or `None` if there is no usable range.
///
/// The limits are the 1st and 99.9th percentiles of the finite values, with the lower limit
/// held at zero or above. `single` says the data was single precision.
#[pyfunction]
pub fn aia_display_limits<'py>(
    data: PyReadonlyArrayDyn<'py, f64>,
    single: bool,
) -> Option<(f64, f64)> {
    let mut finite = values_where(&data, f64::is_finite);
    if finite.is_empty() {
        return None;
    }
    // Python's max(x, 0.0) returns x unless 0.0 is larger, which keeps the sign of a zero.
    let lower = percentile(&mut finite, 1.0, single);
    let vmin = if 0.0 > lower { 0.0 } else { lower };
    let vmax = percentile(&mut finite, 99.9, single);
    (vmin < vmax).then_some((vmin, vmax))
}

/// The display limits for a LASCO image, as `(vmin, vmax)`, or `None` if it has no positive
/// finite value.
///
/// The limits are the 5th and 99.5th percentiles of the positive finite values.
#[pyfunction]
pub fn lasco_display_limits<'py>(
    data: PyReadonlyArrayDyn<'py, f64>,
    single: bool,
) -> Option<(f64, f64)> {
    let mut positive = values_where(&data, |value| value.is_finite() && value > 0.0);
    if positive.is_empty() {
        return None;
    }
    let vmin = percentile(&mut positive, 5.0, single);
    let vmax = percentile(&mut positive, 99.5, single);
    Some((vmin, vmax))
}

/// The symmetric display limit for an HMI magnetogram: the 99.5th percentile of the absolute
/// values, with a limit of zero replaced by one.
///
/// Warns as numpy does when there is no value that is not NaN, and returns NaN.
#[pyfunction]
pub fn magnetogram_limit<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    single: bool,
) -> PyResult<f64> {
    let mut magnitudes: Vec<f64> = values_where(&data, |value| !value.is_nan())
        .into_iter()
        .map(f64::abs)
        .collect();
    if magnitudes.is_empty() {
        let message = if data.as_array().is_empty() {
            c"Mean of empty slice"
        } else {
            c"All-NaN slice encountered"
        };
        PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), message, 1)?;
        return Ok(f64::NAN);
    }
    let limit = percentile(&mut magnitudes, 99.5, single);
    Ok(if limit == 0.0 { 1.0 } else { limit })
}

/// The values of an array that pass a test, in no particular order.
fn values_where<'py>(data: &PyReadonlyArrayDyn<'py, f64>, keep: impl Fn(f64) -> bool) -> Vec<f64> {
    data.as_array()
        .iter()
        .copied()
        .filter(|&value| keep(value))
        .collect()
}

/// numpy's percentile of `values` for `percent` between 0 and 100, using the linear method.
///
/// `values` must hold no NaN and is reordered in place. `single` says the data was single
/// precision, in which case numpy does the interpolation in single precision. The result is NaN
/// when there are no values.
fn percentile(values: &mut [f64], percent: f64, single: bool) -> f64 {
    let count = values.len();
    if count == 0 {
        return f64::NAN;
    }
    let last = count - 1;
    let virtual_index = last as f64 * (percent / 100.0);
    let (lower, upper, gamma) = if virtual_index >= last as f64 {
        (last, last, virtual_index + 1.0)
    } else {
        let below = virtual_index.floor();
        (below as usize, below as usize + 1, virtual_index - below)
    };
    let (low_value, high_value) = order_statistics(values, lower, upper);
    lerp(low_value, high_value, gamma, single)
}

/// The values at two sorted positions, which are the same position when `lower == upper`.
fn order_statistics(values: &mut [f64], lower: usize, upper: usize) -> (f64, f64) {
    let (_, &mut low, right) = values.select_nth_unstable_by(lower, compare);
    let high = if upper == lower {
        low
    } else {
        right.iter().copied().fold(f64::INFINITY, f64::min)
    };
    (low, high)
}

/// Compares two values that are not NaN.
fn compare(first: &f64, second: &f64) -> Ordering {
    first.partial_cmp(second).unwrap_or(Ordering::Equal)
}

/// numpy's linear interpolation between `low` and `high` with weight `gamma`.
///
/// The second formula is used when the weight is at least one half, as numpy does for accuracy.
/// For single precision data numpy converts the weight to single precision too, so every step
/// is rounded to single precision.
fn lerp(low: f64, high: f64, gamma: f64, single: bool) -> f64 {
    if single {
        let (low, high) = (low as f32, high as f32);
        let difference = high - low;
        let value = if gamma >= 0.5 {
            high - difference * (1.0 - gamma) as f32
        } else {
            low + difference * gamma as f32
        };
        f64::from(value)
    } else {
        let difference = high - low;
        if gamma >= 0.5 {
            high - difference * (1.0 - gamma)
        } else {
            low + difference * gamma
        }
    }
}

/// Creates `heliox._core.maps` functions for the factory, the sequence and the sources.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(factory::path_kind, module)?)?;
    module.add_function(wrap_pyfunction!(factory::directory_files, module)?)?;
    module.add_function(wrap_pyfunction!(factory::glob_paths, module)?)?;
    module.add_function(wrap_pyfunction!(sequence::date_order, module)?)?;
    module.add_function(wrap_pyfunction!(instrument_matches, module)?)?;
    module.add_function(wrap_pyfunction!(hmi_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(lasco_colour_table, module)?)?;
    module.add_function(wrap_pyfunction!(nearest_eit_channel, module)?)?;
    module.add_function(wrap_pyfunction!(aia_display_limits, module)?)?;
    module.add_function(wrap_pyfunction!(lasco_display_limits, module)?)?;
    module.add_function(wrap_pyfunction!(magnetogram_limit, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_linearly_between_neighbours() {
        let mut values = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(percentile(&mut values, 50.0, false), 2.5);
        assert_eq!(percentile(&mut values, 0.0, false), 1.0);
        assert_eq!(percentile(&mut values, 100.0, false), 4.0);
    }

    #[test]
    fn percentile_of_one_value_is_that_value() {
        assert_eq!(percentile(&mut [7.5], 99.5, false), 7.5);
    }

    #[test]
    fn percentile_of_nothing_is_nan() {
        assert!(percentile(&mut [], 50.0, false).is_nan());
    }

    #[test]
    fn single_precision_rounds_every_step() {
        // In single precision 1e-8 is lost next to 0.25, and 1 - 1e-8 rounds to 1.
        let low = f64::from(1e-8_f32);
        assert_eq!(lerp(low, 1.0, 0.25, true), 0.25);
        assert_ne!(lerp(low, 1.0, 0.25, false), 0.25);
    }

    #[test]
    fn nearest_channel_rounds_to_the_closest_passband() {
        assert_eq!(nearest_eit_channel(195.0), 193);
        assert_eq!(nearest_eit_channel(94.0), 94);
        assert_eq!(nearest_eit_channel(250.0), 304);
        assert_eq!(nearest_eit_channel(f64::NAN), 94);
    }

    #[test]
    fn nearest_channel_prefers_the_earlier_channel_on_a_tie() {
        assert_eq!(nearest_eit_channel(132.5), 94);
    }

    #[test]
    fn hmi_measurement_reads_the_content_keyword() {
        assert_eq!(hmi_measurement("MAGNETOGRAM"), "magnetogram");
        assert_eq!(hmi_measurement("Line Depth"), "linedepth");
        assert_eq!(hmi_measurement("Continuum Intensity"), "continuum");
        assert_eq!(hmi_measurement("Velocity"), "velocity");
        assert_eq!(hmi_measurement(""), "hmi");
    }

    #[test]
    fn colour_table_depends_on_the_detector() {
        assert_eq!(lasco_colour_table("c3"), "soholasco3");
        assert_eq!(lasco_colour_table("C2"), "soholasco2");
    }

    #[test]
    fn instrument_keywords_match_ignoring_case() {
        assert!(instrument_matches("aia.lev1", "AIA"));
        assert!(!instrument_matches("HMI", "AIA"));
    }
}
