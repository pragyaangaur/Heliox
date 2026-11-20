//! Ordering of map sequences by observation time.

use pyo3::prelude::*;

/// The order in which to visit items so that their Julian dates ascend.
///
/// The sort is stable, so maps with the same date keep the order they were given in. A NaN date
/// sorts after every other date.
#[pyfunction]
pub fn date_order(jd: Vec<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jd.len()).collect();
    order.sort_by(|&first, &second| {
        jd[first]
            .partial_cmp(&jd[second])
            .unwrap_or_else(|| jd[first].is_nan().cmp(&jd[second].is_nan()))
    });
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_ascend_and_equal_dates_keep_their_order() {
        assert_eq!(date_order(vec![3.0, 1.0, 2.0, 1.0]), vec![1, 3, 2, 0]);
    }

    #[test]
    fn nan_dates_come_last() {
        assert_eq!(date_order(vec![f64::NAN, 2.0, 1.0]), vec![2, 1, 0]);
        assert_eq!(date_order(vec![f64::NAN, f64::NAN]), vec![0, 1]);
    }

    #[test]
    fn an_empty_list_has_no_order() {
        assert!(date_order(Vec::new()).is_empty());
    }
}
