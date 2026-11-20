//! Path handling for the `Map` factory: what kind of path it is, the files in a directory,
//! and the files that a glob pattern matches.
//!
//! The glob matcher follows Python's `glob.glob` without recursion. A hidden name (one that
//! starts with a dot) is only matched by a pattern that also starts with a dot.
//!
//! The paths are POSIX paths. The Python factory calls these helpers only on POSIX systems, and
//! uses the standard library on Windows, where the separators and the case rules differ.

use std::fs;

use pyo3::prelude::*;

/// Whether a path contains a glob magic character.
fn has_magic(text: &str) -> bool {
    text.contains(['*', '?', '['])
}

/// The kind of path that a factory argument names.
///
/// Returns `"directory"`, `"file"`, `"pattern"` (a path with glob characters that does not
/// exist) or `"missing"`.
#[pyfunction]
pub fn path_kind(path: &str) -> &'static str {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => "directory",
        Ok(_) => "file",
        Err(_) if has_magic(path) => "pattern",
        Err(_) => "missing",
    }
}

/// The names of the regular files in a directory, sorted.
///
/// Symbolic links are followed, so a link to a file counts as a file.
#[pyfunction]
pub fn directory_files(path: &str) -> PyResult<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(path)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// Every path that matches a glob pattern, sorted.
#[pyfunction]
pub fn glob_paths(pattern: &str) -> Vec<String> {
    let mut paths = glob(pattern, false);
    paths.sort();
    paths
}

/// The paths matching `pathname`, in no particular order.
///
/// `dirs_only` restricts the matches to directories, which is how a directory component
/// of a pattern is expanded. The pathname must contain a magic character.
fn glob(pathname: &str, dirs_only: bool) -> Vec<String> {
    let (dirname, basename) = split(pathname);
    if dirname.is_empty() {
        return glob_in_directory("", basename, dirs_only);
    }
    let directories = if has_magic(dirname) {
        glob(dirname, true)
    } else {
        vec![dirname.to_string()]
    };
    directories
        .iter()
        .flat_map(|directory| {
            glob_in_directory(directory, basename, dirs_only)
                .into_iter()
                .map(move |name| join(directory, &name))
        })
        .collect()
}

/// The names in `directory` that match `basename`, which is a pattern or a literal name.
fn glob_in_directory(directory: &str, basename: &str, dirs_only: bool) -> Vec<String> {
    if has_magic(basename) {
        list_matching(directory, basename, dirs_only)
    } else if basename.is_empty() {
        if fs::metadata(directory).is_ok_and(|metadata| metadata.is_dir()) {
            vec![String::new()]
        } else {
            Vec::new()
        }
    } else if fs::symlink_metadata(join(directory, basename)).is_ok() {
        vec![basename.to_string()]
    } else {
        Vec::new()
    }
}

/// The names in `directory` that match `pattern`, skipping hidden names unless the pattern
/// starts with a dot. A directory that cannot be read has no names.
fn list_matching(directory: &str, pattern: &str, dirs_only: bool) -> Vec<String> {
    let listing = if directory.is_empty() { "." } else { directory };
    let Ok(entries) = fs::read_dir(listing) else {
        return Vec::new();
    };
    let allow_hidden = pattern.starts_with('.');
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let hidden = name.starts_with('.') && !allow_hidden;
            let skipped = dirs_only && !entry.path().is_dir();
            (!hidden && !skipped && fnmatch(&name, pattern)).then_some(name)
        })
        .collect()
}

/// Splits a path into its directory and its final component, as Python's `os.path.split`
/// does for POSIX paths.
fn split(path: &str) -> (&str, &str) {
    let (head, tail) = path.split_at(path.rfind('/').map_or(0, |index| index + 1));
    let trimmed = head.trim_end_matches('/');
    if trimmed.is_empty() {
        (head, tail)
    } else {
        (trimmed, tail)
    }
}

/// Joins a directory and a name, as Python's `os.path.join` does for POSIX paths.
fn join(directory: &str, name: &str) -> String {
    if directory.is_empty() || directory.ends_with('/') {
        format!("{directory}{name}")
    } else {
        format!("{directory}/{name}")
    }
}

/// One element of a compiled shell-style pattern.
enum Token {
    /// A character that matches only itself.
    Literal(char),
    /// `?`, which matches any one character.
    AnyCharacter,
    /// `*`, which matches any run of characters.
    AnyRun,
    /// `[...]` or `[!...]`, given as inclusive character ranges.
    Set {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
}

/// Whether a name matches a shell-style pattern, case-sensitively, as `fnmatch` does.
fn fnmatch(name: &str, pattern: &str) -> bool {
    let name: Vec<char> = name.chars().collect();
    matches_tokens(&compile(pattern), &name)
}

/// Turns a pattern into tokens. An unclosed `[` is an ordinary character.
fn compile(pattern: &str) -> Vec<Token> {
    let characters: Vec<char> = pattern.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        index += 1;
        tokens.push(match character {
            '*' => Token::AnyRun,
            '?' => Token::AnyCharacter,
            '[' => match compile_set(&characters, index) {
                Some((token, next)) => {
                    index = next;
                    token
                }
                None => Token::Literal('['),
            },
            other => Token::Literal(other),
        });
    }
    tokens
}

/// Compiles the set that starts just after a `[`, returning it with the index after `]`.
///
/// Returns `None` when the set has no closing `]`.
fn compile_set(characters: &[char], start: usize) -> Option<(Token, usize)> {
    let mut index = start;
    let negated = characters.get(index) == Some(&'!');
    if negated {
        index += 1;
    }
    let body_start = index;
    if characters.get(index) == Some(&']') {
        index += 1;
    }
    while index < characters.len() && characters[index] != ']' {
        index += 1;
    }
    if index >= characters.len() {
        return None;
    }
    let body = &characters[body_start..index];
    let mut ranges = Vec::new();
    let mut position = 0;
    while position < body.len() {
        if position + 2 < body.len() && body[position + 1] == '-' {
            ranges.push((body[position], body[position + 2]));
            position += 3;
        } else {
            ranges.push((body[position], body[position]));
            position += 1;
        }
    }
    Some((Token::Set { negated, ranges }, index + 1))
}

/// Whether a character list matches a token list in full, backtracking over `*`.
fn matches_tokens(tokens: &[Token], text: &[char]) -> bool {
    match tokens.split_first() {
        None => text.is_empty(),
        Some((Token::AnyRun, rest)) => {
            (0..=text.len()).any(|skip| matches_tokens(rest, &text[skip..]))
        }
        Some((token, rest)) => text.split_first().is_some_and(|(&character, remaining)| {
            single_matches(token, character) && matches_tokens(rest, remaining)
        }),
    }
}

/// Whether a token that matches one character accepts `character`.
///
/// A run never matches a single character here, because `matches_tokens` expands runs.
fn single_matches(token: &Token, character: char) -> bool {
    match token {
        Token::Literal(expected) => *expected == character,
        Token::AnyCharacter => true,
        Token::Set { negated, ranges } => {
            let inside = ranges
                .iter()
                .any(|&(low, high)| low <= character && character <= high);
            inside != *negated
        }
        Token::AnyRun => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_posixpath() {
        assert_eq!(split("data/*.fits"), ("data", "*.fits"));
        assert_eq!(split("/*.fits"), ("/", "*.fits"));
        assert_eq!(split("*.fits"), ("", "*.fits"));
        assert_eq!(split("a//b*"), ("a", "b*"));
        assert_eq!(split("a//"), ("a", ""));
    }

    #[test]
    fn joins_like_posixpath() {
        assert_eq!(join("", "a.fits"), "a.fits");
        assert_eq!(join("data", "a.fits"), "data/a.fits");
        assert_eq!(join("/", "a.fits"), "/a.fits");
    }

    #[test]
    fn star_question_and_sets_match_like_fnmatch() {
        assert!(fnmatch("aia_171.fits", "aia_*.fits"));
        assert!(fnmatch("a.fits", "?.fits"));
        assert!(!fnmatch("ab.fits", "?.fits"));
        assert!(fnmatch("b.fits", "[a-c].fits"));
        assert!(!fnmatch("d.fits", "[a-c].fits"));
        assert!(fnmatch("d.fits", "[!a-c].fits"));
        assert!(fnmatch("[.fits", "[[].fits"));
        assert!(fnmatch("x[.fits", "x[.fits"));
        assert!(!fnmatch("A.fits", "a.fits"));
    }

    #[test]
    fn a_closing_bracket_first_in_a_set_is_literal() {
        assert!(fnmatch("]", "[]]"));
        assert!(!fnmatch("a", "[]]"));
    }
}
