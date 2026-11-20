//! Matching catalogue entries against a flat query.
//!
//! Times are pairs of two-part Julian dates, as astropy stores them. The
//! comparisons below repeat astropy's arithmetic exactly: an ordering test is
//! made on `(a.jd1 - b.jd1) + (a.jd2 - b.jd2)` in the scale both times share,
//! and a difference in seconds is made in TAI. That keeps boundary cases
//! identical to the Python implementation.

use pyo3::prelude::*;

/// Seconds in a day, used to turn a TAI day difference into seconds.
const SECONDS_PER_DAY: f64 = 86400.0;

/// One observation in the catalogue.
#[derive(FromPyObject)]
pub struct Entry {
    /// Start and end, in UTC, as (jd1, jd2) pairs.
    #[pyo3(item)]
    utc_start: (f64, f64),
    #[pyo3(item)]
    utc_end: (f64, f64),
    /// The start in TAI, used for spacing differences.
    #[pyo3(item)]
    tai_start: (f64, f64),
    /// The start as a single Julian date, used for ordering.
    #[pyo3(item)]
    sort_key: f64,
    /// The text fields, already lower-cased by the caller.
    #[pyo3(item)]
    instrument: String,
    #[pyo3(item)]
    source: String,
    #[pyo3(item)]
    detector: String,
    #[pyo3(item)]
    physobs: String,
    /// The level as text, and the wavelength in angstroms (NaN if none).
    #[pyo3(item)]
    level: String,
    #[pyo3(item)]
    wavelength: f64,
}

/// The conditions of a flat query. Every condition in a list must hold.
#[derive(FromPyObject)]
pub struct Query {
    /// Each window as (start, end) UTC pairs. The entry must overlap each one.
    #[pyo3(item)]
    times: Vec<((f64, f64), (f64, f64))>,
    #[pyo3(item)]
    instruments: Vec<String>,
    #[pyo3(item)]
    sources: Vec<String>,
    #[pyo3(item)]
    detectors: Vec<String>,
    #[pyo3(item)]
    physobs: Vec<String>,
    #[pyo3(item)]
    levels: Vec<String>,
    /// Each range as (minimum, maximum) angstroms, inclusive.
    #[pyo3(item)]
    wavelengths: Vec<(f64, f64)>,
    /// The minimum spacing between returned records in seconds, if any.
    #[pyo3(item)]
    minimum_spacing: Option<f64>,
}

impl Query {
    /// Does one entry satisfy every condition in the query?
    fn accepts(&self, entry: &Entry) -> bool {
        self.times
            .iter()
            .all(|&(start, end)| overlaps(entry.utc_start, entry.utc_end, start, end))
            && all_equal(&self.instruments, &entry.instrument)
            && all_equal(&self.sources, &entry.source)
            && all_equal(&self.detectors, &entry.detector)
            && all_equal(&self.physobs, &entry.physobs)
            && all_equal(&self.levels, &entry.level)
            && self
                .wavelengths
                .iter()
                .all(|&(minimum, maximum)| in_range(entry.wavelength, minimum, maximum))
    }
}

/// Returns the indices of the entries that match, ordered by start time.
///
/// The indices refer to `entries`. When `minimum_spacing` is given, an entry is
/// dropped if it starts less than that many seconds after the last kept entry.
#[pyfunction]
pub fn select_entries(entries: Vec<Entry>, query: Query) -> Vec<usize> {
    let mut kept: Vec<usize> = (0..entries.len())
        .filter(|&index| query.accepts(&entries[index]))
        .collect();
    kept.sort_by(|&a, &b| entries[a].sort_key.total_cmp(&entries[b].sort_key));
    match query.minimum_spacing {
        Some(minimum) => thin(&entries, &kept, minimum),
        None => kept,
    }
}

/// Keeps the entries that start at least `minimum` seconds after the last kept one.
fn thin(entries: &[Entry], ordered: &[usize], minimum: f64) -> Vec<usize> {
    let mut kept = Vec::new();
    let mut last: Option<(f64, f64)> = None;
    for &index in ordered {
        let start = entries[index].tai_start;
        let spaced = match last {
            None => true,
            Some(previous) => seconds_between(start, previous) >= minimum,
        };
        if spaced {
            kept.push(index);
            last = Some(start);
        }
    }
    kept
}

/// The difference `later - earlier` in seconds, for TAI pairs.
fn seconds_between(later: (f64, f64), earlier: (f64, f64)) -> f64 {
    ((later.0 - earlier.0) + (later.1 - earlier.1)) * SECONDS_PER_DAY
}

/// Do the intervals `[start, end]` and `[query_start, query_end]` overlap?
fn overlaps(
    start: (f64, f64),
    end: (f64, f64),
    query_start: (f64, f64),
    query_end: (f64, f64),
) -> bool {
    let ends_before = ordering(end, query_start) < 0.0;
    let starts_after = ordering(start, query_end) > 0.0;
    !(ends_before || starts_after)
}

/// The sign of `a - b` in astropy's comparison arithmetic.
fn ordering(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0) + (a.1 - b.1)
}

/// Is `value` inside the inclusive range? NaN is never inside.
fn in_range(value: f64, minimum: f64, maximum: f64) -> bool {
    value.is_finite() && minimum <= value && value <= maximum
}

/// Does every item in `wanted` equal `actual`? An empty list always passes.
fn all_equal(wanted: &[String], actual: &str) -> bool {
    wanted.iter().all(|value| value == actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOON: (f64, f64) = (2456594.0, 0.5);

    #[test]
    fn touching_intervals_overlap() {
        let end = NOON;
        assert!(overlaps((2456593.0, 0.0), end, NOON, (2456595.0, 0.0)));
    }

    #[test]
    fn disjoint_intervals_do_not_overlap() {
        let start = (2456600.0, 0.0);
        let end = (2456601.0, 0.0);
        assert!(!overlaps(start, end, (2456590.0, 0.0), (2456599.0, 0.0)));
    }

    #[test]
    fn a_nan_wavelength_is_never_in_range() {
        assert!(!in_range(f64::NAN, 0.0, f64::INFINITY));
        assert!(in_range(171.0, 170.0, 172.0));
        assert!(in_range(171.0, 171.0, 171.0));
        assert!(!in_range(193.0, 170.0, 172.0));
    }

    #[test]
    fn spacing_is_in_seconds() {
        let minute = seconds_between((0.0, 60.0 / SECONDS_PER_DAY), (0.0, 0.0));
        assert!((minute - 60.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_list_of_conditions_passes() {
        assert!(all_equal(&[], "anything"));
    }
}
