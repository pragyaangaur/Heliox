//! Reads the `# key: value` comment lines that some archives put at the top of a CSV file.

use std::fs::File;
use std::io::{BufRead, BufReader};

use pyo3::exceptions::PyUnicodeDecodeError;
use pyo3::prelude::*;

/// Whether `c` is a space in the sense of Python's `str.strip`.
///
/// That is Unicode whitespace, which Rust's `char::is_whitespace` also covers, plus the file
/// and group separators U+001C to U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Splits text into lines the way Python's universal newlines do. `\n`, `\r` and `\r\n` each
/// end a line, and empty text has no lines.
fn lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    let mut done = text.is_empty();
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        match rest.find(['\r', '\n']) {
            Some(end) => {
                let line = &rest[..end];
                rest = if rest[end..].starts_with("\r\n") {
                    &rest[end + 2..]
                } else {
                    &rest[end + 1..]
                };
                done = rest.is_empty();
                Some(line)
            }
            None => {
                done = true;
                Some(rest)
            }
        }
    })
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

/// Returns the `key: value` pairs of the comment lines at the top of a file, in file order.
///
/// Reading stops at the first line that does not start with `#`. Only the lines read so far are
/// decoded, so bytes after the header are never checked. Keys that repeat are returned as often
/// as they occur, and the caller decides which one wins.
#[pyfunction]
pub fn header_pairs(py: Python<'_>, path: &str) -> PyResult<Vec<(String, String)>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut chunk = Vec::new();
    let mut pairs = Vec::new();
    loop {
        chunk.clear();
        if reader.read_until(b'\n', &mut chunk)? == 0 {
            return Ok(pairs);
        }
        let text = match std::str::from_utf8(&chunk) {
            Ok(text) => text,
            Err(error) => {
                let decode_error = PyUnicodeDecodeError::new_utf8(py, &chunk, error)?;
                return Err(PyErr::from_value(decode_error.into_any()));
            }
        };
        for line in lines(text) {
            if !line.starts_with('#') {
                return Ok(pairs);
            }
            pairs.extend(pair(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{lines, pair};

    #[test]
    fn every_newline_style_ends_a_line() {
        let collected: Vec<&str> = lines("a\nb\r\nc\rd").collect();
        assert_eq!(collected, ["a", "b", "c", "d"]);
    }

    #[test]
    fn empty_text_has_no_lines_and_a_trailing_newline_adds_none() {
        assert_eq!(lines("").count(), 0);
        assert_eq!(lines("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(lines("\n").collect::<Vec<_>>(), [""]);
    }

    #[test]
    fn pairs_strip_hashes_and_spaces() {
        assert_eq!(
            pair("#  instrume :  XRS  "),
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
