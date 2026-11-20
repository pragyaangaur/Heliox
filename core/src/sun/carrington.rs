//! Carrington rotation numbers, and the inverse map from a rotation number back
//! to a time.

use super::ephemeris::python_mod;
use super::rootfind::{self, BrentError};

/// The Julian date at which Carrington rotation 1 began.
const EPOCH_JD: f64 = 2_398_167.4;
/// The mean synodic rotation period adopted by Carrington, in days.
const SYNODIC_PERIOD_DAYS: f64 = 27.2753;
/// The absolute tolerance on the Julian date of a rotation time, in days.
const TIME_XTOL: f64 = 1e-8;
/// How far either side of the mean-period guess the root is bracketed, in days.
const BRACKET_DAYS: f64 = 3.0;

/// The Carrington rotation number, including its fractional part, from a TT
/// Julian date and the disc-centre Carrington longitude L0 in degrees.
///
/// The longitude gives the exact fraction. The mean period gives a first guess
/// for the integer part, which is nudged by one when the two disagree across a
/// rotation boundary.
pub fn rotation_number(jd: f64, l0: f64) -> f64 {
    let estimate = (jd - EPOCH_JD) / SYNODIC_PERIOD_DAYS + 1.0;
    let estimate_frac = python_mod(estimate, 1.0);
    let estimate_int = (estimate - estimate_frac).floor();
    let exact_frac = 1.0 - l0 / 360.0;
    let nudged_int = estimate_int + (estimate_frac - exact_frac).round_ties_even();
    nudged_int + exact_frac
}

/// The TT Julian date of each Carrington rotation number in `targets`.
///
/// `rotation_numbers` gives the rotation number of each Julian date it is
/// passed, in the same order. Each date is found by a root search on
/// `rotation_number(jd) - target`, started from a bracket around the date that
/// the mean period predicts. Returns the dates in the order of `targets`.
pub fn rotation_times<E>(
    targets: &[f64],
    mut rotation_numbers: impl FnMut(&[f64]) -> Result<Vec<f64>, E>,
) -> Result<Vec<f64>, BrentError<E>> {
    let guesses: Vec<f64> = targets
        .iter()
        .map(|&target| (target - 1.0) * SYNODIC_PERIOD_DAYS + EPOCH_JD)
        .collect();
    let lower: Vec<f64> = guesses.iter().map(|&g| g - BRACKET_DAYS).collect();
    let upper: Vec<f64> = guesses.iter().map(|&g| g + BRACKET_DAYS).collect();

    rootfind::solve_all(
        &lower,
        &upper,
        TIME_XTOL,
        rootfind::DEFAULT_RTOL,
        |owners, jds| {
            let numbers = rotation_numbers(jds)?;
            Ok(numbers
                .iter()
                .zip(owners)
                .map(|(&number, &owner)| number - targets[owner])
                .collect())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_rotation_starts_at_the_epoch() {
        // At the epoch the mean-period estimate is exactly 1 and L0 is 360.
        let number = rotation_number(EPOCH_JD, 360.0);
        assert!((number - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_rotation_time_is_found_from_a_linear_rotation_number() {
        // For this stand-in the root is exactly EPOCH + (target - 1) * period.
        let jds = rotation_times(&[5.0, 2143.0], |jds| {
            Ok::<_, ()>(
                jds.iter()
                    .map(|&jd| 1.0 + (jd - EPOCH_JD) / SYNODIC_PERIOD_DAYS)
                    .collect(),
            )
        })
        .expect("the brackets straddle the roots");
        assert!((jds[0] - (EPOCH_JD + 4.0 * SYNODIC_PERIOD_DAYS)).abs() < 1e-7);
        assert!((jds[1] - (EPOCH_JD + 2142.0 * SYNODIC_PERIOD_DAYS)).abs() < 1e-7);
    }
}
