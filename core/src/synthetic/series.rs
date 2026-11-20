//! Per-sample kernels for the synthetic GOES light curve and sunspot series.
//!
//! As in `image.rs`, each function repeats the numpy expression it replaces,
//! with the same operations in the same order.

use std::f64::consts::PI;

use ndarray::ArrayView1;

use super::image::{clip_below, clip_both};

/// The slowly varying background of the long-wavelength channel.
///
/// The sample times are `i * cadence_minutes` for each entry of `normals`,
/// which holds one standard normal draw per sample.
pub fn xray_background(normals: ArrayView1<f64>, cadence_minutes: f64) -> Vec<f64> {
    let last_minute = normals.len().saturating_sub(1) as f64 * cadence_minutes;
    let period = if last_minute == 0.0 { 1.0 } else { last_minute };
    normals
        .iter()
        .enumerate()
        .map(|(index, &z)| {
            let minute = index as f64 * cadence_minutes;
            let wave = (2.0 * PI * minute / period).sin();
            clip_below(1.5e-7 * (1.0 + 0.3 * wave + 0.05 * z), 1e-9)
        })
        .collect()
}

/// Adds one flare to the long-wavelength channel.
///
/// The flare rises with time constant `rise` before its peak and decays with
/// time constant `decay` after it, scaled to `peak_flux`.
pub fn xray_flare(
    curve: ArrayView1<f64>,
    cadence_minutes: f64,
    peak_time: f64,
    peak_flux: f64,
    rise: f64,
    decay: f64,
) -> Vec<f64> {
    curve
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let offset = index as f64 * cadence_minutes - peak_time;
            let profile = if offset < 0.0 {
                (offset / rise).exp()
            } else {
                (-offset / decay).exp()
            };
            value + peak_flux * profile
        })
        .collect()
}

/// The short and long GOES channels, derived from the long-channel flux.
///
/// Returns `(xrsa, xrsb)`, each clipped to its floor. `normals` holds one
/// standard normal draw per sample for the short channel.
pub fn xray_channels(curve: ArrayView1<f64>, normals: ArrayView1<f64>) -> (Vec<f64>, Vec<f64>) {
    let short = curve
        .iter()
        .zip(normals.iter())
        .map(|(&long, &z)| {
            let ratio = 0.02 + 0.25 * clip_both((long.log10() + 7.0) / 3.0, 0.0, 1.0);
            let flux = long * ratio;
            clip_below(flux * (1.0 + 0.05 * z), 1e-10)
        })
        .collect();
    let long = curve.iter().map(|&flux| clip_below(flux, 1e-9)).collect();
    (short, long)
}

/// The monthly sunspot number over an eleven year cycle with scatter.
///
/// `normals` holds one standard normal draw per month.
pub fn sunspot_numbers(normals: ArrayView1<f64>) -> Vec<f64> {
    normals
        .iter()
        .enumerate()
        .map(|(index, &z)| {
            let phase = (index as f64 / 12.0) % 11.0;
            let cycle = if phase < 4.0 {
                (PI * phase / 8.0).sin()
            } else {
                (PI * (4.0 + (phase - 4.0) * 4.0 / 7.0) / 8.0).sin()
            };
            clip_below(120.0 * cycle + 12.0 * z, 0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn background_stays_near_the_quiet_sun_level() {
        let normals = array![0.0, 0.0, 0.0, 0.0];
        let background = xray_background(normals.view(), 1.0);
        assert_eq!(background.len(), 4);
        assert!(background.iter().all(|&v| (1e-8..1e-6).contains(&v)));
    }

    #[test]
    fn a_single_sample_background_uses_a_unit_period() {
        let background = xray_background(array![0.0].view(), 1.0);
        assert_eq!(background, vec![1.5e-7]);
    }

    #[test]
    fn flare_adds_its_peak_flux_at_peak_time() {
        let curve = array![1.0e-7, 1.0e-7, 1.0e-7];
        let flared = xray_flare(curve.view(), 1.0, 1.0, 1.0e-5, 2.0, 4.0);
        assert!((flared[1] - (1.0e-7 + 1.0e-5)).abs() < 1e-12);
        assert!(flared[1] > flared[0]);
    }

    #[test]
    fn channels_respect_their_floors() {
        let (short, long) = xray_channels(array![1e-12, 1e-6].view(), array![0.0, 0.0].view());
        assert!(short.iter().all(|&v| v >= 1e-10));
        assert!(long.iter().all(|&v| v >= 1e-9));
    }

    #[test]
    fn sunspot_numbers_are_never_negative() {
        let numbers = sunspot_numbers(array![-10.0, -10.0, 0.0, 0.0].view());
        assert!(numbers.iter().all(|&v| v >= 0.0));
    }
}
