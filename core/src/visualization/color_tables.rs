//! The numeric data behind the heliox colour tables.
//!
//! Each table is three curves, one per colour channel, sampled at evenly
//! spaced points on the unit interval. The matplotlib colormap objects are
//! built from these curves in Python.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// The number of samples in each colour table.
const TABLE_SIZE: usize = 256;

/// The AIA passbands, in angstroms, with the exponents of their red, green and
/// blue curves. A smaller exponent brightens a channel, so the channel with the
/// smallest exponent dominates the appearance of the passband.
///
/// The order is the order in which the tables are registered.
const AIA_CHANNELS: [(i64, [f64; 3]); 10] = [
    (1600, [0.75, 0.75, 1.00]),
    (1700, [0.75, 1.00, 1.50]),
    (4500, [0.70, 0.90, 1.30]),
    (94, [1.50, 1.00, 0.60]),
    (131, [1.00, 0.60, 0.90]),
    (171, [1.20, 0.70, 1.60]),
    (193, [0.80, 1.10, 1.60]),
    (211, [0.90, 1.30, 0.80]),
    (304, [0.70, 1.50, 1.60]),
    (335, [1.40, 1.10, 0.70]),
];

/// `count` evenly spaced points from 0 to 1 inclusive, as `numpy.linspace`
/// computes them.
fn linspace(count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![0.0];
    }
    let step = 1.0 / (count - 1) as f64;
    let mut points: Vec<f64> = (0..count).map(|i| i as f64 * step).collect();
    points[count - 1] = 1.0;
    points
}

/// The curve `x ** exponent` on the table grid, clipped to the unit interval.
fn power_curve(exponent: f64) -> Vec<f64> {
    linspace(TABLE_SIZE)
        .into_iter()
        .map(|x| x.powf(exponent).clamp(0.0, 1.0))
        .collect()
}

/// The blue-white-red curves: red rises, blue falls, and green peaks at the
/// middle of the table.
fn bipolar_curves() -> [Vec<f64>; 3] {
    let x = linspace(TABLE_SIZE);
    let red = x.iter().map(|&x| (2.0 * x).clamp(0.0, 1.0)).collect();
    let green = x.iter().map(|&x| 1.0 - (2.0 * x - 1.0).abs()).collect();
    let blue = x
        .iter()
        .map(|&x| (2.0 * (1.0 - x)).clamp(0.0, 1.0))
        .collect();
    [red, green, blue]
}

/// The AIA passbands that have colour tables, in angstroms.
#[pyfunction]
pub fn aia_wavelengths() -> Vec<i64> {
    AIA_CHANNELS.iter().map(|(angstrom, _)| *angstrom).collect()
}

/// The red, green and blue exponents of an AIA passband, or `None` if the
/// wavelength in angstroms is not an AIA channel.
#[pyfunction]
pub fn aia_exponents(angstrom: i64) -> Option<(f64, f64, f64)> {
    AIA_CHANNELS
        .iter()
        .find(|(channel, _)| *channel == angstrom)
        .map(|(_, [red, green, blue])| (*red, *green, *blue))
}

/// A curve of the form `x ** exponent` on the 256-point table grid.
#[pyfunction]
pub fn gamma_curve<'py>(py: Python<'py>, exponent: f64) -> Bound<'py, PyArray1<f64>> {
    power_curve(exponent).into_pyarray(py)
}

/// The straight line from 0 to 1 on the 256-point table grid.
#[pyfunction]
pub fn linear_curve<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    linspace(TABLE_SIZE).into_pyarray(py)
}

/// The red, green and blue curves of the blue-white-red table.
#[pyfunction]
pub fn bipolar_channels<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
    bipolar_curves()
        .into_iter()
        .map(|curve| curve.into_pyarray(py))
        .collect()
}

/// Pairs each value with its position on the unit interval, as the
/// `(x, y0, y1)` rows that matplotlib takes for one colour channel.
#[pyfunction]
pub fn segment_data<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let values = values.as_array();
    let positions = linspace(values.len());
    let flat: Vec<f64> = positions
        .iter()
        .zip(values.iter())
        .flat_map(|(&x, &value)| [x, value, value])
        .collect();
    Array2::from_shape_vec((positions.len(), 3), flat)
        .expect("three values per row")
        .into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_ends_exactly_on_one() {
        let points = linspace(TABLE_SIZE);
        assert_eq!(points[0], 0.0);
        assert_eq!(points[TABLE_SIZE - 1], 1.0);
        assert!((points[1] - 1.0 / 255.0).abs() < 1e-18);
    }

    #[test]
    fn power_curve_with_exponent_one_is_the_ramp() {
        assert_eq!(power_curve(1.0), linspace(TABLE_SIZE));
    }

    #[test]
    fn bipolar_green_peaks_in_the_middle() {
        let [_, green, _] = bipolar_curves();
        assert!((green[TABLE_SIZE / 2] - 1.0).abs() < 0.01);
        assert_eq!(green[0], 0.0);
        assert_eq!(green[TABLE_SIZE - 1], 0.0);
    }

    #[test]
    fn aia_lookup_matches_the_table() {
        assert_eq!(aia_exponents(171), Some((1.2, 0.7, 1.6)));
        assert_eq!(aia_exponents(500), None);
        assert_eq!(aia_wavelengths().len(), AIA_CHANNELS.len());
    }
}
