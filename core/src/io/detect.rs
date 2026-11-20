//! File type detection for `heliox.io.file_tools`.
//!
//! The magic number is checked first, so that a file with the wrong extension
//! is still recognised. The extension is used only when no magic number matches.

use pyo3::prelude::*;

/// Leading bytes that identify each supported format.
const MAGIC_NUMBERS: [(&[u8], &str); 2] =
    [(b"SIMPLE  =", "fits"), (b"\x00\x00\x00\x0cjP  ", "jp2")];

/// Lower-case file suffixes that identify each supported format.
const EXTENSIONS: [(&str, &str); 4] = [
    (".fits", "fits"),
    (".fit", "fits"),
    (".fts", "fits"),
    (".fits.gz", "fits"),
];

/// Returns the type name for the leading bytes of a file, or its suffix.
///
/// `head` is the first bytes of the file and `suffix` is the lower-case
/// concatenation of its suffixes, such as `".fits.gz"`. Returns `None` if
/// neither identifies a supported format.
#[pyfunction]
pub fn detect_filetype(head: &[u8], suffix: &str) -> Option<&'static str> {
    let by_magic = MAGIC_NUMBERS
        .iter()
        .find(|(magic, _)| head.starts_with(magic))
        .map(|&(_, name)| name);
    by_magic.or_else(|| {
        EXTENSIONS
            .iter()
            .find(|(extension, _)| suffix.ends_with(extension))
            .map(|&(_, name)| name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_number_wins_over_extension() {
        assert_eq!(detect_filetype(b"SIMPLE  =    T", ".txt"), Some("fits"));
        assert_eq!(
            detect_filetype(b"\x00\x00\x00\x0cjP  ", ".fits"),
            Some("jp2")
        );
    }

    #[test]
    fn extension_is_the_fallback() {
        assert_eq!(detect_filetype(b"junk", ".fits.gz"), Some("fits"));
        assert_eq!(detect_filetype(b"", ".fit"), Some("fits"));
        assert_eq!(detect_filetype(b"junk", ".unknown"), None);
    }
}
