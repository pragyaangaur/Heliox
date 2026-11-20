//! FITS header text handling for `heliox.io._fits`.
//!
//! The Python caller reads the keyword and the string form of every card. The
//! card values themselves can be any Python object, so they stay in Python and
//! Rust only returns the positions of the cards to keep.
//!
//! Python's `str.strip`, `str.split` and `str.isspace` treat the characters
//! U+001C to U+001F as whitespace, and Rust's `char::is_whitespace` does not.
//! The helpers here follow Python so that the results are identical.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The keyword of a COMMENT card.
const COMMENT: &str = "COMMENT";

/// The keyword of a HISTORY card.
const HISTORY: &str = "HISTORY";

/// Recognised spellings of wavelength units, mapped onto their canonical names.
const WAVEUNITS: [(&str, &str); 11] = [
    ("angstrom", "Angstrom"),
    ("angstroms", "Angstrom"),
    ("a", "Angstrom"),
    ("nm", "nm"),
    ("nanometer", "nm"),
    ("nanometers", "nm"),
    ("um", "um"),
    ("micron", "um"),
    ("micrometer", "um"),
    ("mm", "mm"),
    ("m", "m"),
];

/// Whether Python's `str.isspace` accepts a character.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The text with leading and trailing whitespace removed, as `str.strip` does.
fn py_strip(text: &str) -> &str {
    text.trim_matches(is_py_space)
}

/// The whitespace-separated words of a text, as `str.split()` gives them.
fn py_words(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_py_space).filter(|word| !word.is_empty())
}

/// The canonical unit for a recognised spelling, which must be lower case.
fn lookup_waveunit(spelling: &str) -> Option<&'static str> {
    WAVEUNITS
        .iter()
        .find(|(known, _)| *known == spelling)
        .map(|&(_, canonical)| canonical)
}

/// Joins comment or history lines with newlines, or returns `None` if there are none.
fn join_lines(lines: &[String]) -> Option<String> {
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The ordinary cards to keep, then the joined COMMENT and HISTORY text.
type CollapsedHeader = (Vec<(String, usize)>, Option<String>, Option<String>);

/// Sorts header cards into ordinary keywords and collapsed commentary.
///
/// `keywords[i]` is the keyword of card `i` and `texts[i]` is the string form
/// of its value. The result holds the keyword and index of each ordinary card
/// in file order, which Python then reads from the original card list, and the
/// stripped COMMENT and HISTORY lines joined with newlines. Blank cards are
/// dropped. A commentary entry is `None` when the header has no such card.
///
/// # Errors
///
/// Returns `ValueError` if the two lists differ in length.
#[pyfunction]
pub fn collapse_header(keywords: Vec<String>, texts: Vec<String>) -> PyResult<CollapsedHeader> {
    if keywords.len() != texts.len() {
        return Err(PyValueError::new_err(
            "keywords and texts must have the same length",
        ));
    }

    let mut entries = Vec::new();
    let mut comments = Vec::new();
    let mut history = Vec::new();
    for (index, (keyword, text)) in keywords.into_iter().zip(texts).enumerate() {
        if keyword == COMMENT {
            comments.push(py_strip(&text).to_string());
        } else if keyword == HISTORY {
            history.push(py_strip(&text).to_string());
        } else if !keyword.is_empty() {
            entries.push((keyword, index));
        }
    }
    Ok((entries, join_lines(&comments), join_lines(&history)))
}

/// Works out the canonical name of a wavelength unit.
///
/// A `raw` value from the WAVEUNIT keyword is checked first. If it is
/// unrecognised, it is returned stripped but otherwise unchanged. Without a
/// `raw` value, the words of the `comment` text are searched in order, with
/// commas treated as spaces, and the first recognised one is returned.
#[pyfunction]
pub fn extract_waveunit(raw: Option<&str>, comment: Option<&str>) -> Option<String> {
    match raw {
        Some(raw) => {
            let text = py_strip(raw);
            let canonical = lookup_waveunit(&text.to_lowercase());
            Some(canonical.map_or_else(|| text.to_string(), str::to_string))
        }
        None => comment.and_then(|text| {
            let lower = text.to_lowercase().replace(',', " ");
            let found = py_words(&lower).find_map(lookup_waveunit);
            found.map(str::to_string)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_follows_python_whitespace() {
        assert_eq!(py_strip("\x1cnm\x1f"), "nm");
        assert_eq!(py_strip("  \u{a0}Å \t"), "Å");
    }

    #[test]
    fn words_split_on_runs_of_whitespace() {
        let words: Vec<&str> = py_words("  a\tb\x1c\x1dc  ").collect();
        assert_eq!(words, ["a", "b", "c"]);
    }

    #[test]
    fn waveunit_from_the_keyword_is_canonical() {
        assert_eq!(
            extract_waveunit(Some("Angstroms"), None).as_deref(),
            Some("Angstrom")
        );
        assert_eq!(
            extract_waveunit(Some(" micron "), None).as_deref(),
            Some("um")
        );
        assert_eq!(
            extract_waveunit(Some("furlongs"), None).as_deref(),
            Some("furlongs")
        );
        assert_eq!(extract_waveunit(Some(""), None).as_deref(), Some(""));
    }

    #[test]
    fn waveunit_from_a_comment_takes_the_first_match() {
        let comment = Some("wavelength given in nanometer, and later a");
        assert_eq!(extract_waveunit(None, comment).as_deref(), Some("nm"));
        assert_eq!(extract_waveunit(None, Some("no unit here")), None);
    }

    #[test]
    fn collapse_keeps_order_and_joins_commentary() {
        let keywords = ["A", "COMMENT", "", "HISTORY", "B", "COMMENT", "A"]
            .map(String::from)
            .to_vec();
        let texts = ["1", " first ", "", "did it", "2", "second", "3"]
            .map(String::from)
            .to_vec();
        let (entries, comment, history) = collapse_header(keywords, texts).unwrap();
        assert_eq!(
            entries,
            [
                ("A".to_string(), 0),
                ("B".to_string(), 4),
                ("A".to_string(), 6)
            ]
        );
        assert_eq!(comment.as_deref(), Some("first\nsecond"));
        assert_eq!(history.as_deref(), Some("did it"));
    }

    #[test]
    fn collapse_reports_missing_commentary_as_none() {
        let (entries, comment, history) =
            collapse_header(vec!["A".to_string()], vec!["1".to_string()]).unwrap();
        assert_eq!(entries, [("A".to_string(), 0)]);
        assert!(comment.is_none() && history.is_none());
    }
}
