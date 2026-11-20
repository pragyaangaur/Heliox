//! Recognises the time strings that `heliox.time.parse_time` accepts.
//!
//! A format is compiled once into tokens. A string is accepted by a format in
//! two steps, which mirror the two checks the Python code used to make:
//!
//! 1. The shape check. Every field must have a fixed set of digit shapes and
//!    the literal characters must match exactly. This is the anchored regular
//!    expression that was built for each format.
//! 2. The strptime check. The tokens are matched the way `datetime.strptime`
//!    matches them, with its own patterns and its order of alternatives. The
//!    first match in that order must use the whole string, and the resulting
//!    fields must form a real date with a second below 60.
//!
//! Strings that fail either check return `None`, and astropy then gets the
//! final say. That is how leap seconds such as `23:59:60` and times such as
//! `24:00:00` are handled.
//!
//! The Python caller maps Unicode decimal digits to ASCII before calling this
//! module and passes the positions of the mapped characters. The shape check
//! accepts any decimal digit, as `\d` does. Strptime's patterns accept only
//! ASCII digits in some positions, so the mapped characters are tracked.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The English month abbreviations. `%b` accepts them in any case.
const MONTH_ABBREVIATIONS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

/// A time component that a format directive reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    /// `%f`, the fractional second.
    Fraction,
    /// `%b`, the month as a three-letter name.
    MonthAbbrev,
    /// `%j`, the day of the year.
    DayOfYear,
    /// `%z`, a UTC offset.
    Offset,
}

/// One element of a compiled format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token {
    Literal(char),
    Field(Field),
}

/// Compiles a format string such as `%Y-%m-%dT%H:%M:%S`.
///
/// A `%` followed by a supported directive is a field. Any other `%`, including
/// one at the very end, is a literal.
fn compile(format: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = format.chars().collect();
    let mut tokens = Vec::with_capacity(chars.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '%' && index + 1 < chars.len() {
            let field = match chars[index + 1] {
                'Y' => Field::Year,
                'm' => Field::Month,
                'd' => Field::Day,
                'H' => Field::Hour,
                'M' => Field::Minute,
                'S' => Field::Second,
                'f' => Field::Fraction,
                'b' => Field::MonthAbbrev,
                'j' => Field::DayOfYear,
                'z' => Field::Offset,
                other => {
                    return Err(format!("Unsupported directive '%{other}' in {format:?}"));
                }
            };
            tokens.push(Token::Field(field));
            index += 2;
        } else {
            tokens.push(Token::Literal(chars[index]));
            index += 1;
        }
    }
    Ok(tokens)
}

/// One position of a pattern, which matches one character.
#[derive(Clone, Copy, Debug)]
enum Class {
    /// The exact ASCII character.
    Exact(char),
    /// An ASCII character from the first to the last, inclusive, as in `[0-5]`.
    Range(char, char),
    /// A decimal digit, as `\d` matches. This includes Unicode digits.
    Digit,
    /// An ASCII letter, as in `[a-zA-Z]`.
    Letter,
    /// A plus or a minus sign.
    Sign,
}

/// One character of the text.
#[derive(Clone, Copy, Debug)]
struct Character {
    value: char,
    /// `true` if the character was a Unicode decimal digit before the Python
    /// caller mapped it to ASCII.
    unicode_digit: bool,
}

impl Class {
    fn accepts(self, character: Character) -> bool {
        let Character {
            value,
            unicode_digit,
        } = character;
        match self {
            Class::Exact(expected) => !unicode_digit && value == expected,
            Class::Range(first, last) => !unicode_digit && (first..=last).contains(&value),
            Class::Digit => value.is_ascii_digit(),
            Class::Letter => !unicode_digit && value.is_ascii_alphabetic(),
            Class::Sign => !unicode_digit && matches!(value, '+' | '-'),
        }
    }
}

/// The alternatives of the shape check for a field, longest first.
///
/// These are the anchored patterns `\d{1,2}`, `\d+` and so on, with the
/// strptime-only restrictions left out. Fraction and month-name fields have no
/// entry here because they are handled in `candidates`.
fn shape_alternatives(field: Field) -> &'static [&'static [Class]] {
    use Class::{Digit, Exact, Sign};
    match field {
        Field::Year => &[&[Digit, Digit, Digit, Digit]],
        Field::Month | Field::Day | Field::Hour | Field::Minute | Field::Second => {
            &[&[Digit, Digit], &[Digit]]
        }
        Field::DayOfYear => &[&[Digit, Digit, Digit], &[Digit, Digit], &[Digit]],
        Field::Offset => &[
            &[Exact('Z')],
            &[Sign, Digit, Digit, Exact(':'), Digit, Digit],
            &[Sign, Digit, Digit, Digit, Digit],
        ],
        Field::Fraction | Field::MonthAbbrev => &[],
    }
}

/// The alternatives of the strptime regular expression for a field, in the
/// order that CPython's `_strptime` lists them.
fn strptime_alternatives(field: Field) -> &'static [&'static [Class]] {
    use Class::{Digit, Exact, Range, Sign};
    match field {
        Field::Year => &[&[Digit, Digit, Digit, Digit]],
        // 1[0-2]|0[1-9]|[1-9]
        Field::Month => &[
            &[Exact('1'), Range('0', '2')],
            &[Exact('0'), Range('1', '9')],
            &[Range('1', '9')],
        ],
        // 3[0-1]|[1-2]\d|0[1-9]|[1-9]
        Field::Day => &[
            &[Exact('3'), Range('0', '1')],
            &[Range('1', '2'), Digit],
            &[Exact('0'), Range('1', '9')],
            &[Range('1', '9')],
        ],
        // 2[0-3]|[0-1]\d|\d
        Field::Hour => &[
            &[Exact('2'), Range('0', '3')],
            &[Range('0', '1'), Digit],
            &[Digit],
        ],
        // [0-5]\d|\d
        Field::Minute => &[&[Range('0', '5'), Digit], &[Digit]],
        // 6[0-1]|[0-5]\d|\d
        Field::Second => &[
            &[Exact('6'), Range('0', '1')],
            &[Range('0', '5'), Digit],
            &[Digit],
        ],
        // 36[0-6]|3[0-5]\d|[1-2]\d\d|0[1-9]\d|00[1-9]|[1-9]\d|0[1-9]|[1-9]
        Field::DayOfYear => &[
            &[Exact('3'), Exact('6'), Range('0', '6')],
            &[Exact('3'), Range('0', '5'), Digit],
            &[Range('1', '2'), Digit, Digit],
            &[Exact('0'), Range('1', '9'), Digit],
            &[Exact('0'), Exact('0'), Range('1', '9')],
            &[Range('1', '9'), Digit],
            &[Exact('0'), Range('1', '9')],
            &[Range('1', '9')],
        ],
        // [+-]\d\d:?[0-5]\d|Z, where the optional seconds never occur because the
        // shape check has already ended the text.
        Field::Offset => &[
            &[Exact('Z')],
            &[Sign, Digit, Digit, Exact(':'), Range('0', '5'), Digit],
            &[Sign, Digit, Digit, Range('0', '5'), Digit],
        ],
        Field::Fraction | Field::MonthAbbrev => &[],
    }
}

/// The value of a run of digits.
fn digits_value(characters: &[Character]) -> i64 {
    characters.iter().fold(0, |acc, c| {
        acc * 10 + i64::from(c.value.to_digit(10).unwrap_or(0))
    })
}

/// The alternatives that a field can match at `pos`, as `(length, value)` pairs.
///
/// Strptime mode returns them in the order strptime tries them, and applies its
/// value limits. Shape mode returns every shape that fits, which is what the
/// anchored shape regex needs.
fn candidates(field: Field, text: &[Character], pos: usize, strptime: bool) -> Vec<(usize, i64)> {
    let rest = text.get(pos..).unwrap_or(&[]);
    match field {
        Field::Fraction => {
            let run = rest.iter().take_while(|c| c.value.is_ascii_digit()).count();
            if !strptime {
                return (1..=run).map(|len| (len, 0)).collect();
            }
            // Strptime reads at most six ASCII digits, and the Python code cut any
            // longer fraction down to six before strptime saw it. Digits past the
            // sixth are dropped, so the whole run is consumed.
            let kept = &rest[..run.min(6)];
            if run == 0 || kept.iter().any(|c| c.unicode_digit) {
                return Vec::new();
            }
            let value = digits_value(kept) * 10_i64.pow(6 - kept.len() as u32);
            vec![(run, value)]
        }
        Field::MonthAbbrev => {
            let Some(name) = rest.get(..3) else {
                return Vec::new();
            };
            if !name.iter().all(|c| Class::Letter.accepts(*c)) {
                return Vec::new();
            }
            if !strptime {
                return vec![(3, 0)];
            }
            MONTH_ABBREVIATIONS
                .iter()
                .position(|month| {
                    month
                        .chars()
                        .zip(name)
                        .all(|(expected, c)| c.value.to_ascii_lowercase() == expected)
                })
                .map_or_else(Vec::new, |index| vec![(3, index as i64 + 1)])
        }
        Field::Offset => {
            let alternatives = if strptime {
                strptime_alternatives(field)
            } else {
                shape_alternatives(field)
            };
            alternatives
                .iter()
                .filter_map(|alternative| {
                    let span = rest.get(..alternative.len())?;
                    if !alternative
                        .iter()
                        .zip(span)
                        .all(|(class, c)| class.accepts(*c))
                    {
                        return None;
                    }
                    if span.len() == 1 {
                        return Some((1, 0));
                    }
                    let hours = digits_value(&span[1..3]);
                    let minutes = digits_value(&span[span.len() - 2..]);
                    let seconds = hours * 3600 + minutes * 60;
                    let sign = if span[0].value == '-' { -1 } else { 1 };
                    // datetime.timezone refuses offsets of a whole day or more.
                    if strptime && seconds >= 86_400 {
                        return None;
                    }
                    Some((span.len(), sign * seconds))
                })
                .collect()
        }
        _ => {
            let alternatives = if strptime {
                strptime_alternatives(field)
            } else {
                shape_alternatives(field)
            };
            alternatives
                .iter()
                .filter_map(|alternative| {
                    let span = rest.get(..alternative.len())?;
                    alternative
                        .iter()
                        .zip(span)
                        .all(|(class, c)| class.accepts(*c))
                        .then(|| (span.len(), digits_value(span)))
                })
                .collect()
        }
    }
}

/// Returns `true` if the whole text fits the shape of `tokens`.
fn shape_matches(tokens: &[Token], text: &[Character], pos: usize) -> bool {
    let Some((first, rest)) = tokens.split_first() else {
        return pos == text.len();
    };
    match *first {
        Token::Literal(expected) => {
            text.get(pos)
                .is_some_and(|c| Class::Exact(expected).accepts(*c))
                && shape_matches(rest, text, pos + 1)
        }
        Token::Field(field) => candidates(field, text, pos, false)
            .into_iter()
            .any(|(len, _)| shape_matches(rest, text, pos + len)),
    }
}

/// The values that strptime collects from a successful match.
struct Fields {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    microsecond: i64,
    julian: Option<i64>,
    offset: Option<i64>,
}

impl Default for Fields {
    /// The defaults that `datetime.strptime` uses for fields the format leaves out.
    fn default() -> Self {
        Self {
            year: 1900,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            microsecond: 0,
            julian: None,
            offset: None,
        }
    }
}

impl Fields {
    fn store(&mut self, field: Field, value: i64) {
        match field {
            Field::Year => self.year = value,
            Field::Month | Field::MonthAbbrev => self.month = value,
            Field::Day => self.day = value,
            Field::Hour => self.hour = value,
            Field::Minute => self.minute = value,
            Field::Second => self.second = value,
            Field::Fraction => self.microsecond = value,
            Field::DayOfYear => self.julian = Some(value),
            Field::Offset => self.offset = Some(value),
        }
    }
}

/// Matches `tokens` from `pos` the way the strptime regular expression does.
///
/// Returns the position where the first successful match ends. Alternatives are
/// tried in order, and the first complete match wins, as in `re.match`.
fn strptime_from(
    tokens: &[Token],
    text: &[Character],
    pos: usize,
    fields: &mut Fields,
) -> Option<usize> {
    let Some((first, rest)) = tokens.split_first() else {
        return Some(pos);
    };
    match *first {
        Token::Literal(expected) => {
            if text
                .get(pos)
                .is_some_and(|c| Class::Exact(expected).accepts(*c))
            {
                strptime_from(rest, text, pos + 1, fields)
            } else {
                None
            }
        }
        Token::Field(field) => {
            for (len, value) in candidates(field, text, pos, true) {
                fields.store(field, value);
                if let Some(end) = strptime_from(rest, text, pos + len, fields) {
                    return Some(end);
                }
            }
            None
        }
    }
}

/// Days from 1970-01-01 to the given proleptic Gregorian date.
///
/// This is Howard Hinnant's `days_from_civil` algorithm.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The proleptic Gregorian date for a count of days from 1970-01-01.
///
/// This is Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Turns the collected fields into a calendar date and time, the way
/// `datetime.strptime` does, and rejects anything `datetime` would refuse.
fn to_naive_iso(fields: &Fields) -> Option<String> {
    if !(1..=9999).contains(&fields.year) || !(0..=59).contains(&fields.second) {
        return None;
    }
    let (year, month, day) = match fields.julian {
        Some(julian) => civil_from_days(days_from_civil(fields.year, 1, 1) + julian - 1),
        None if fields.day <= days_in_month(fields.year, fields.month) => {
            (fields.year, fields.month, fields.day)
        }
        None => return None,
    };
    if year > 9999 {
        return None;
    }
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}",
        fields.hour, fields.minute, fields.second, fields.microsecond
    ))
}

/// Parses one string with one compiled format.
///
/// Returns the naive ISO time, and the UTC offset in seconds if the format had `%z`.
fn parse_with(tokens: &[Token], text: &[Character]) -> Option<(String, Option<i32>)> {
    if !shape_matches(tokens, text, 0) {
        return None;
    }
    let mut fields = Fields::default();
    let end = strptime_from(tokens, text, 0, &mut fields)?;
    if end != text.len() {
        return None;
    }
    let iso = to_naive_iso(&fields)?;
    let offset = fields.offset.map(|seconds| seconds as i32);
    Some((iso, offset))
}

/// Builds the characters of a mapped string. `unicode_digits` lists the positions
/// of the characters that were Unicode digits before they were mapped.
fn characters(text: &str, unicode_digits: &[usize]) -> Vec<Character> {
    text.chars()
        .enumerate()
        .map(|(index, value)| Character {
            value,
            unicode_digit: unicode_digits.contains(&index),
        })
        .collect()
}

/// The formats that `heliox.time.parse_time` tries, in order.
#[pyclass(module = "heliox._core.time")]
pub struct TimeFormats {
    formats: Vec<Vec<Token>>,
}

#[pymethods]
impl TimeFormats {
    /// Compiles the formats. Raises `ValueError` for a directive that is not supported.
    #[new]
    fn new(formats: Vec<String>) -> PyResult<Self> {
        let formats = formats
            .iter()
            .map(|format| compile(format))
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(Self { formats })
    }

    /// Parses each string with the first format that accepts it.
    ///
    /// Each item is a pair of the text, with Unicode digits mapped to ASCII, and
    /// the positions of the characters that were mapped. Each result is `None` if
    /// no format accepts the string. Otherwise it is a pair of the naive ISO time
    /// with microseconds, and the UTC offset in seconds, or `None` if the string
    /// had no offset.
    fn parse(&self, texts: Vec<(String, Vec<usize>)>) -> Vec<Option<(String, Option<i32>)>> {
        texts
            .iter()
            .map(|(text, unicode_digits)| {
                let text = characters(text, unicode_digits);
                self.formats
                    .iter()
                    .find_map(|tokens| parse_with(tokens, &text))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(format: &str, text: &str) -> Option<(String, Option<i32>)> {
        parse_with(&compile(format).unwrap(), &characters(text, &[]))
    }

    #[test]
    fn unsupported_directive_is_an_error() {
        assert!(compile("%Q").is_err());
        assert!(compile("%Y%%").is_err());
        assert_eq!(compile("%Y%").unwrap().len(), 2);
    }

    #[test]
    fn iso_with_space_and_fraction() {
        assert_eq!(
            parse_one("%Y-%m-%d %H:%M:%S.%f", "2013-10-28 14:30:00.5"),
            Some(("2013-10-28T14:30:00.500000".to_string(), None))
        );
    }

    #[test]
    fn fraction_keeps_six_digits() {
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S.%f", "2013-10-28T14:30:00.1234567891"),
            Some(("2013-10-28T14:30:00.123456".to_string(), None))
        );
    }

    #[test]
    fn leap_seconds_and_hour_24_are_left_to_astropy() {
        assert_eq!(parse_one("%Y-%m-%dT%H:%M:%S", "2016-12-31T23:59:60"), None);
        assert_eq!(parse_one("%Y-%m-%dT%H:%M:%S", "2013-10-28T24:00:00"), None);
    }

    #[test]
    fn offsets_are_reported_in_seconds() {
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00+05:30"),
            Some(("2013-10-28T14:30:00.000000".to_string(), Some(19_800)))
        );
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00-0100"),
            Some(("2013-10-28T14:30:00.000000".to_string(), Some(-3600)))
        );
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00Z"),
            Some(("2013-10-28T14:30:00.000000".to_string(), Some(0)))
        );
    }

    #[test]
    fn offsets_outside_the_strptime_range_are_rejected() {
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00+24:00"),
            None
        );
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00+01:60"),
            None
        );
        assert_eq!(
            parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00+01:00:00"),
            None
        );
        assert!(parse_one("%Y-%m-%dT%H:%M:%S%z", "2013-10-28T14:30:00+23:59").is_some());
    }

    #[test]
    fn day_of_year_rolls_over_and_is_bounded() {
        assert_eq!(
            parse_one("%Y-%j", "2013-301"),
            Some(("2013-10-28T00:00:00.000000".to_string(), None))
        );
        assert_eq!(
            parse_one("%Y-%j", "2013-366"),
            Some(("2014-01-01T00:00:00.000000".to_string(), None))
        );
        assert_eq!(
            parse_one("%Y-%j", "2012-366"),
            Some(("2012-12-31T00:00:00.000000".to_string(), None))
        );
        assert_eq!(parse_one("%Y-%j", "2013-000"), None);
        assert_eq!(parse_one("%Y-%j", "9999-366"), None);
    }

    #[test]
    fn impossible_calendar_dates_are_rejected() {
        assert_eq!(parse_one("%Y-%m-%d", "2013-02-29"), None);
        assert!(parse_one("%Y-%m-%d", "2012-02-29").is_some());
        assert_eq!(parse_one("%Y-%m-%d", "0000-01-01"), None);
        assert_eq!(parse_one("%Y-%m-%d", "2013-13-01"), None);
    }

    #[test]
    fn month_names_ignore_case() {
        assert_eq!(
            parse_one("%d-%b-%Y", "28-OCT-2013"),
            Some(("2013-10-28T00:00:00.000000".to_string(), None))
        );
        assert_eq!(parse_one("%d-%b-%Y", "28-Foo-2013"), None);
    }

    #[test]
    fn strptime_refuses_a_month_that_is_not_followed_by_the_separator() {
        // The shape allows "13". Strptime reads "1" and then fails on the "3".
        assert_eq!(parse_one("%Y-%m-%d", "2013-13-1"), None);
        assert!(parse_one("%Y-%m-%d", "2013-1-5").is_some());
    }

    #[test]
    fn lowercase_t_is_refused_by_the_shape_check() {
        assert_eq!(parse_one("%Y-%m-%dT%H:%M:%S", "2013-10-28t14:30:00"), None);
    }

    #[test]
    fn unicode_digits_follow_the_strptime_patterns() {
        // Month and day tens digits are ASCII only in strptime, so a Unicode second
        // digit is refused after a "1" in the month.
        let mapped = |text: &str, positions: &[usize]| characters(text, positions);
        let month = compile("%Y-%m-%d").unwrap();
        assert_eq!(
            parse_with(&month, &mapped("2013-10-28", &[])).map(|r| r.0),
            Some("2013-10-28T00:00:00.000000".to_string())
        );
        assert_eq!(parse_with(&month, &mapped("2013-1020-28", &[8])), None);
        // A Unicode digit in the second place of a day is accepted, as `[1-2]\d` allows.
        assert!(parse_with(&month, &mapped("2013-10-28", &[9])).is_some());
        // The year accepts any decimal digit.
        assert!(parse_with(&month, &mapped("2013-10-28", &[0])).is_some());
    }

    #[test]
    fn days_round_trip_through_civil_dates() {
        for days in [-719_162, -1, 0, 1, 10_957, 19_000, 2_932_896] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }
}
