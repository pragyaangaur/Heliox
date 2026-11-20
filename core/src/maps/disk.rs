//! Tests of helioprojective points against the solar disc.

use ndarray::Zip;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use super::check_same_shape;

/// Whether a helioprojective point, given in arcseconds, projects onto the disc.
///
/// This is a two-dimensional test, so points behind the Sun count as on the disc.
fn on_disk(tx: f64, ty: f64, radius: f64) -> bool {
    (tx * tx + ty * ty).sqrt() < radius
}

/// Which points lie on the solar disc.
///
/// `tx`, `ty` and `radius` are in arcseconds and must have the same shape, so the caller
/// broadcasts the radius first.
#[pyfunction]
pub fn on_disk_mask<'py>(
    py: Python<'py>,
    tx: PyReadonlyArrayDyn<'py, f64>,
    ty: PyReadonlyArrayDyn<'py, f64>,
    radius: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let (tx, ty, radius) = (tx.as_array(), ty.as_array(), radius.as_array());
    check_same_shape(tx.shape(), ty.shape())?;
    check_same_shape(tx.shape(), radius.shape())?;
    Ok(Zip::from(&tx)
        .and(&ty)
        .and(&radius)
        .map_collect(|&tx, &ty, &radius| on_disk(tx, ty, radius))
        .into_pyarray(py))
}

/// The extent of the points that are marked on the disc, as `(tx_min, tx_max, ty_min, ty_max)`.
///
/// Returns `None` if no point is marked. All three inputs must have the same shape.
#[pyfunction]
pub fn disk_extent<'py>(
    tx: PyReadonlyArrayDyn<'py, f64>,
    ty: PyReadonlyArrayDyn<'py, f64>,
    mask: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<Option<(f64, f64, f64, f64)>> {
    let (tx, ty, mask) = (tx.as_array(), ty.as_array(), mask.as_array());
    check_same_shape(tx.shape(), ty.shape())?;
    check_same_shape(tx.shape(), mask.shape())?;
    let mut extent: Option<(f64, f64, f64, f64)> = None;
    Zip::from(&tx)
        .and(&ty)
        .and(&mask)
        .for_each(|&tx, &ty, &marked| {
            if marked {
                extent = Some(match extent {
                    None => (tx, tx, ty, ty),
                    Some((x0, x1, y0, y1)) => (x0.min(tx), x1.max(tx), y0.min(ty), y1.max(ty)),
                });
            }
        });
    Ok(extent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_on_the_disc_inside_its_radius() {
        assert!(on_disk(3.0, 4.0, 6.0));
        assert!(!on_disk(3.0, 4.0, 5.0));
        assert!(!on_disk(f64::NAN, 0.0, 5.0));
    }
}
