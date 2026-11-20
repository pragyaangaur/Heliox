//! Solar coordinate families and the FITS keywords that name them.
//!
//! A family is one of the four solar frame kinds that heliox writes to, or
//! reads from, FITS headers. The Python side passes families by name.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The solar coordinate families that heliox can describe in a FITS header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Helioprojective,
    Stonyhurst,
    Carrington,
    Heliocentric,
}

impl Family {
    /// Parses a family from the lower-case name that the Python side uses.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "helioprojective" => Some(Self::Helioprojective),
            "stonyhurst" => Some(Self::Stonyhurst),
            "carrington" => Some(Self::Carrington),
            "heliocentric" => Some(Self::Heliocentric),
            _ => None,
        }
    }

    /// The lower-case name of the family.
    pub fn name(self) -> &'static str {
        match self {
            Self::Helioprojective => "helioprojective",
            Self::Stonyhurst => "stonyhurst",
            Self::Carrington => "carrington",
            Self::Heliocentric => "heliocentric",
        }
    }

    /// The family that a `CTYPE` keyword belongs to, or `None` if it is not solar.
    ///
    /// Only the first four characters matter, and they are compared without
    /// regard to case, as FITS readers do.
    pub fn from_ctype(ctype: &str) -> Option<Self> {
        let prefix = ctype
            .chars()
            .take(4)
            .collect::<String>()
            .to_ascii_uppercase();
        match prefix.as_str() {
            "HPLN" | "HPLT" => Some(Self::Helioprojective),
            "HGLN" | "HGLT" => Some(Self::Stonyhurst),
            "CRLN" | "CRLT" => Some(Self::Carrington),
            "SOLX" | "SOLY" => Some(Self::Heliocentric),
            _ => None,
        }
    }

    /// The `CTYPE` prefixes of the longitude and latitude axes.
    fn axis_prefixes(self) -> (&'static str, &'static str) {
        match self {
            Self::Helioprojective => ("HPLN", "HPLT"),
            Self::Stonyhurst => ("HGLN", "HGLT"),
            Self::Carrington => ("CRLN", "CRLT"),
            Self::Heliocentric => ("SOLX", "SOLY"),
        }
    }

    /// The unit that both axes of the family use in a FITS header.
    pub fn axis_unit(self) -> &'static str {
        match self {
            Self::Helioprojective => "arcsec",
            Self::Heliocentric => "m",
            Self::Stonyhurst | Self::Carrington => "deg",
        }
    }
}

fn parse_family(name: &str) -> PyResult<Family> {
    Family::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown solar family: {name}")))
}

/// The one family that the solar `CTYPE` keywords belong to.
///
/// Returns `None` if none of the keywords is solar, or if they name more than
/// one family. The result is the family name, such as `"stonyhurst"`.
#[pyfunction]
pub fn solar_family(ctypes: Vec<String>) -> Option<&'static str> {
    let mut found = None;
    for family in ctypes.iter().filter_map(|ctype| Family::from_ctype(ctype)) {
        match found {
            None => found = Some(family),
            Some(previous) if previous != family => return None,
            Some(_) => {}
        }
    }
    found.map(Family::name)
}

/// The `CTYPE` keywords of the longitude and latitude axes for a family and projection.
#[pyfunction]
pub fn ctype_names(family: &str, projection: &str) -> PyResult<(String, String)> {
    let (lon, lat) = parse_family(family)?.axis_prefixes();
    Ok((format!("{lon}-{projection}"), format!("{lat}-{projection}")))
}

/// The `CUNIT` value that both axes of a family use.
#[pyfunction]
pub fn axis_unit(family: &str) -> PyResult<&'static str> {
    Ok(parse_family(family)?.axis_unit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctype_prefix_ignores_case_and_suffix() {
        assert_eq!(
            Family::from_ctype("hpln-tan"),
            Some(Family::Helioprojective)
        );
        assert_eq!(Family::from_ctype("CRLT-CAR"), Some(Family::Carrington));
        assert_eq!(Family::from_ctype("RA---TAN"), None);
        assert_eq!(Family::from_ctype("HG"), None);
    }

    #[test]
    fn one_family_is_found_and_mixed_families_are_rejected() {
        let solar = |names: &[&str]| solar_family(names.iter().map(|s| s.to_string()).collect());
        assert_eq!(solar(&["HPLN-TAN", "HPLT-TAN"]), Some("helioprojective"));
        assert_eq!(solar(&["RA---TAN", "DEC--TAN"]), None);
        assert_eq!(solar(&["HPLN-TAN", "HGLT-CAR"]), None);
        assert_eq!(solar(&[]), None);
    }

    #[test]
    fn ctype_names_and_units_follow_the_family() {
        let (lon, lat) = ctype_names("carrington", "CAR").unwrap();
        assert_eq!((lon.as_str(), lat.as_str()), ("CRLN-CAR", "CRLT-CAR"));
        assert_eq!(axis_unit("helioprojective").unwrap(), "arcsec");
        assert_eq!(axis_unit("heliocentric").unwrap(), "m");
        assert_eq!(axis_unit("stonyhurst").unwrap(), "deg");
    }
}
