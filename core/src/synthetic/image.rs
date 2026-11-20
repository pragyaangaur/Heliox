//! Per-pixel kernels for the synthetic solar images.
//!
//! Each function repeats one array expression from the Python generator, with
//! the same operations in the same order, so the output matches numpy bit for
//! bit. The clipping helpers follow numpy's rules, including the sign of zero.

use ndarray::{Array2, ArrayView2, Zip};

/// A Gaussian feature added to an image: `amplitude * exp(-r^2 / denominator)`,
/// where `r^2` is the squared distance from `centre`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaussian {
    /// Column and row of the peak, in pixels.
    pub centre: (f64, f64),
    /// Twice the squared width, as in `2 * width**2`.
    pub denominator: f64,
    /// The height of the peak.
    pub amplitude: f64,
}

/// A coronal streamer: a broad enhancement at one position angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Streamer {
    /// The position angle of the streamer, in radians.
    pub angle: f64,
    /// Twice the squared angular width, as in `2 * width**2`.
    pub denominator: f64,
    /// The peak enhancement.
    pub strength: f64,
}

/// `np.clip(value, low, None)`: the value, or `low` when the value is below it.
pub fn clip_below(value: f64, low: f64) -> f64 {
    if value > low || value.is_nan() {
        value
    } else {
        low
    }
}

/// `np.clip(value, low, high)` for numbers.
pub fn clip_both(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// The largest absolute value in an array, or one if that value is zero.
///
/// This is the `np.abs(a).max() or 1.0` that normalises the noise fields.
fn peak_or_one(values: ArrayView2<f64>) -> f64 {
    let peak = values
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs()));
    if peak == 0.0 {
        1.0
    } else {
        peak
    }
}

/// The fractional distance from the disc centre, in units of the solar radius.
///
/// `centre` is the column and row of the disc centre in pixels, and `radius`
/// is the solar radius in pixels.
pub fn radial_grid(shape: (usize, usize), centre: (f64, f64), radius: f64) -> Array2<f64> {
    Array2::from_shape_fn(shape, |(row, col)| {
        (col as f64 - centre.0).hypot(row as f64 - centre.1) / radius
    })
}

/// The position angle of every pixel about the centre, in radians.
pub fn position_angle_grid(shape: (usize, usize), centre: (f64, f64)) -> Array2<f64> {
    Array2::from_shape_fn(shape, |(row, col)| {
        (row as f64 - centre.1).atan2(col as f64 - centre.0)
    })
}

/// Coronal EUV emission: brighter at the limb on the disc, decaying above it.
pub fn euv_emission(radial: ArrayView2<f64>) -> Array2<f64> {
    radial.mapv(|r| {
        if r <= 1.0 {
            let clipped = clip_both(r, 0.0, 0.995);
            let path_length = 1.0 / (1.0 - clipped * clipped).sqrt();
            100.0 * clip_both(path_length, 1.0, 4.0)
        } else {
            400.0 * (-(r - 1.0) / 0.13).exp()
        }
    })
}

/// Visible emission: the limb darkening profile scaled on the disc, zero off it.
pub fn visible_emission(radial: ArrayView2<f64>, limb_profile: ArrayView2<f64>) -> Array2<f64> {
    Zip::from(radial)
        .and(limb_profile)
        .map_collect(|&r, &profile| if r <= 1.0 { 1000.0 * profile } else { 0.0 })
}

/// The sum of a set of Gaussian features, added in the order given.
pub fn gaussian_sum(shape: (usize, usize), gaussians: &[Gaussian]) -> Array2<f64> {
    let mut field = Array2::<f64>::zeros(shape);
    for gaussian in gaussians {
        for ((row, col), value) in field.indexed_iter_mut() {
            let dx = col as f64 - gaussian.centre.0;
            let dy = row as f64 - gaussian.centre.1;
            let exponent = -(dx * dx + dy * dy) / gaussian.denominator;
            *value += gaussian.amplitude * exponent.exp();
        }
    }
    field
}

/// Multiplies the image by the granulation pattern, on the disc only.
///
/// The noise is first divided by its peak absolute value.
pub fn granulate(
    image: ArrayView2<f64>,
    noise: ArrayView2<f64>,
    radial: ArrayView2<f64>,
) -> Array2<f64> {
    let peak = peak_or_one(noise);
    Zip::from(image)
        .and(noise)
        .and(radial)
        .map_collect(|&pixel, &value, &r| {
            let on_disc = if r <= 1.0 { 1.0 } else { 0.0 };
            let granulation = value / peak;
            pixel * (1.0 + 0.08 * granulation * on_disc)
        })
}

/// Adds the quiet-Sun network to the field, scaled to `amplitude` gauss at
/// its peak, and keeps only the disc.
pub fn network_field(
    field: ArrayView2<f64>,
    network: ArrayView2<f64>,
    radial: ArrayView2<f64>,
    amplitude: f64,
) -> Array2<f64> {
    let factor = amplitude / peak_or_one(network);
    Zip::from(field)
        .and(network)
        .and(radial)
        .map_collect(|&value, &noise, &r| {
            if r <= 1.0 {
                value + noise * factor
            } else {
                0.0
            }
        })
}

/// Adds photon noise, whose spread grows with the square root of the signal,
/// and clips the result at zero.
///
/// `normals` holds standard normal draws, one per pixel.
pub fn photon_noise(image: ArrayView2<f64>, normals: ArrayView2<f64>) -> Array2<f64> {
    Zip::from(image).and(normals).map_collect(|&pixel, &z| {
        let scale = clip_below(pixel, 1.0).sqrt() * 0.3;
        clip_below(pixel + scale * z, 0.0)
    })
}

/// The summed enhancement of the coronal streamers at each pixel.
pub fn streamer_field(position_angle: ArrayView2<f64>, streamers: &[Streamer]) -> Array2<f64> {
    let mut field = Array2::<f64>::zeros(position_angle.dim());
    for streamer in streamers {
        for (value, &angle) in field.iter_mut().zip(position_angle.iter()) {
            let offset = angle - streamer.angle;
            // The black_box stops the compiler fusing sin and cos into one sincos
            // call, which rounds differently from numpy's separate calls.
            let separation = offset.sin().atan2(std::hint::black_box(offset).cos());
            let exponent = -(separation * separation) / streamer.denominator;
            *value += streamer.strength * exponent.exp();
        }
    }
    field
}

/// White-light coronal brightness with the occulter applied and noise added.
///
/// `enhancement` is the streamer field and `normals` holds standard normal
/// draws, one per pixel.
pub fn coronagraph_brightness(
    radial: ArrayView2<f64>,
    enhancement: ArrayView2<f64>,
    normals: ArrayView2<f64>,
    occulter: f64,
) -> Array2<f64> {
    Zip::from(radial)
        .and(enhancement)
        .and(normals)
        .map_collect(|&r, &streamers, &z| {
            let falloff = 1e4 * clip_below(r, 1.0).powf(-3.5);
            let brightness = falloff * (1.0 + streamers);
            let brightness = if r < occulter { 0.0 } else { brightness };
            let noise = 2.0 * z;
            let outside = if r >= occulter { 1.0 } else { 0.0 };
            clip_below(brightness + noise * outside, 0.0)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn clip_below_keeps_values_above_the_floor() {
        assert_eq!(clip_below(2.5, 1.0), 2.5);
        assert_eq!(clip_below(-3.0, 1.0), 1.0);
        assert!(clip_below(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn clip_below_returns_the_floor_for_negative_zero() {
        assert!(!clip_below(-0.0, 0.0).is_sign_negative());
    }

    #[test]
    fn clip_both_respects_both_bounds() {
        assert_eq!(clip_both(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(clip_both(2.0, 0.0, 1.0), 1.0);
        assert!(clip_both(-0.0, 0.0, 1.0).is_sign_negative());
    }

    #[test]
    fn radial_grid_is_zero_at_the_centre_and_one_at_the_limb() {
        let grid = radial_grid((5, 5), (2.0, 2.0), 2.0);
        assert_eq!(grid[[2, 2]], 0.0);
        assert_eq!(grid[[2, 4]], 1.0);
        assert_eq!(grid[[0, 2]], 1.0);
    }

    #[test]
    fn gaussian_sum_peaks_at_its_centre() {
        let field = gaussian_sum(
            (3, 3),
            &[Gaussian {
                centre: (1.0, 1.0),
                denominator: 2.0,
                amplitude: 5.0,
            }],
        );
        assert_eq!(field[[1, 1]], 5.0);
        assert!(field[[0, 0]] < field[[1, 1]]);
    }

    #[test]
    fn photon_noise_never_goes_negative() {
        let image = array![[0.0, 4.0], [100.0, 1.0]];
        let normals = array![[-50.0, 0.0], [0.0, -2.0]];
        assert!(photon_noise(image.view(), normals.view())
            .iter()
            .all(|&v| v >= 0.0));
    }

    #[test]
    fn euv_emission_is_positive_everywhere() {
        let radial = array![[0.0, 0.5, 1.0, 1.5]];
        assert!(euv_emission(radial.view()).iter().all(|&v| v > 0.0));
    }
}
