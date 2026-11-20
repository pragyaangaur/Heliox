//! The text that `heliox.util.decorators.deprecated` shows to users.

use pyo3::prelude::*;

/// Builds the warning text for a deprecated object.
///
/// An empty `message` gives the default sentence, which names the object and
/// the release. A non-empty `alternative` adds a sentence that points to its
/// replacement.
#[pyfunction]
pub fn deprecation_text(name: &str, since: &str, message: &str, alternative: &str) -> String {
    let mut text = if message.is_empty() {
        format!("{name} is deprecated as of heliox {since} and will be removed.")
    } else {
        message.to_string()
    };
    if !alternative.is_empty() {
        text = format!("{text} Use {alternative} instead.");
    }
    text
}

/// Appends a `deprecated` note to an object's docstring.
#[pyfunction]
pub fn deprecated_docstring(doc: &str, since: &str, text: &str) -> String {
    format!("{doc}\n\n.. deprecated:: {since}\n    {text}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_names_the_object() {
        assert_eq!(
            deprecation_text("old", "0.2", "", ""),
            "old is deprecated as of heliox 0.2 and will be removed."
        );
    }

    #[test]
    fn alternative_is_appended_to_a_custom_message() {
        assert_eq!(
            deprecation_text("old", "0.2", "Gone.", "new"),
            "Gone. Use new instead."
        );
    }

    #[test]
    fn docstring_gets_a_deprecated_note() {
        assert_eq!(
            deprecated_docstring("Doc.", "0.2", "Text."),
            "Doc.\n\n.. deprecated:: 0.2\n    Text.\n"
        );
    }
}
