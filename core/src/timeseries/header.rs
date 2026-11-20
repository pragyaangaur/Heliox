//! Reads the `# key: value` comment lines that some archives put at the top of a CSV file.

use pyo3::prelude::*;

/// Whether `c` is a space in the sense of Python's `str.strip`.
///
/// That is Unicode whitespace, which Rust's `char::is_whitespace` also covers, plus the file
/// and group separators U+001C to U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Reads the `key: value` pair from one comment line, or `None` if the line has no colon.
///
/// The leading `#` characters are dropped, the rest is stripped, and the pair is split at the
/// first colon. Each side is stripped again.
fn pair(line: &str) -> Option<(String, String)> {
    let body = line.trim_start_matches('#').trim_matches(is_space);
    let (key, value) = body.split_once(':')?;
    Some((
        key.trim_matches(is_space).to_string(),
        value.trim_matches(is_space).to_string(),
    ))
}

/// Returns the `key: value` pairs of the comment lines at the top of a text file, in file order.
///
/// `lines` is an iterable of text lines, such as an open file. Python does the decoding and the
/// newline handling, so the result follows the platform encoding exactly as `open` does. Reading
/// stops at the first line that does not start with `#`, and later lines are never requested.
/// Keys that repeat are returned as often as they occur, and the caller decides which one wins.
#[pyfunction]
pub fn header_pairs(lines: &Bound<'_, PyAny>) -> PyResult<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    for line in lines.try_iter()? {
        let line: String = line?.extract()?;
        if !line.starts_with('#') {
            break;
        }
        pairs.extend(pair(&line));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::pair;

    #[test]
    fn pairs_strip_hashes_and_spaces() {
        assert_eq!(
            pair("#  instrume :  XRS  \n"),
            Some(("instrume".to_string(), "XRS".to_string()))
        );
    }

    #[test]
    fn the_value_keeps_later_colons() {
        assert_eq!(
            pair("# DATE-OBS: 2013-10-28T00:00:00"),
            Some(("DATE-OBS".to_string(), "2013-10-28T00:00:00".to_string()))
        );
    }

    #[test]
    fn lines_without_a_colon_give_no_pair() {
        assert_eq!(pair("# just a comment"), None);
        assert_eq!(pair("#"), None);
    }

    #[test]
    fn unit_and_file_separators_are_stripped_like_python() {
        assert_eq!(
            pair("#\u{1c}key\u{1f}:\u{1c}value"),
            Some(("key".to_string(), "value".to_string()))
        );
    }
}
