import pytest

import astropy.units as u

from heliox.net import attrs as a
from heliox.net.attrs import AttrAnd, AttrOr, and_, or_, to_sum_of_products
from heliox.time import TimeRange


# ---------------------------------------------------------------------------
# Individual attributes
# ---------------------------------------------------------------------------
def test_time_from_two_times():
    attribute = a.Time("2013-10-28", "2013-10-29")
    assert attribute.start.isot == "2013-10-28T00:00:00.000"
    assert attribute.end.isot == "2013-10-29T00:00:00.000"


def test_time_from_a_timerange():
    window = TimeRange("2013-10-28", "2013-10-29")
    assert a.Time(window).start == window.start


def test_time_orders_its_ends():
    attribute = a.Time("2013-10-29", "2013-10-28")
    assert attribute.start < attribute.end


def test_time_needs_an_end():
    with pytest.raises(ValueError, match="both a start and an end"):
        a.Time("2013-10-28")


def test_time_range_property():
    attribute = a.Time("2013-10-28", "2013-10-29")
    assert isinstance(attribute.time_range, TimeRange)
    assert attribute.time_range.days.to_value(u.day) == pytest.approx(1)


def test_time_near():
    attribute = a.Time("2013-10-28", "2013-10-29", near="2013-10-28T12:00")
    assert attribute.near.isot == "2013-10-28T12:00:00.000"


def test_time_contains():
    attribute = a.Time("2013-10-28", "2013-10-29")
    assert "2013-10-28T12:00" in attribute
    assert "2013-11-01" not in attribute
    assert "not a time" not in attribute


def test_wavelength_single_value():
    attribute = a.Wavelength(171 * u.angstrom)
    assert attribute.start == attribute.end == 171 * u.angstrom


def test_wavelength_range():
    attribute = a.Wavelength(100 * u.angstrom, 200 * u.angstrom)
    assert 171 * u.angstrom in attribute
    assert 300 * u.angstrom not in attribute


def test_wavelength_converts_units():
    attribute = a.Wavelength(17.1 * u.nm)
    assert attribute.start.to_value(u.angstrom) == pytest.approx(171)


def test_wavelength_needs_a_length():
    with pytest.raises(ValueError, match="units of length"):
        a.Wavelength(171 * u.s)


def test_simple_attrs():
    assert str(a.Instrument("AIA")) == "AIA"
    assert repr(a.Source("SDO")) == "<Source('SDO')>"
    assert a.Detector("C2").value == "C2"
    assert a.Physobs("intensity").value == "intensity"


def test_level_accepts_numbers_and_strings():
    assert a.Level(1.5).value == 1.5
    assert a.Level("1.5").value == "1.5"


def test_level_rejects_other_types():
    with pytest.raises(ValueError, match="number or a string"):
        a.Level([1, 2])


def test_sample_needs_a_positive_duration():
    assert a.Sample(1 * u.minute).value == 1 * u.minute
    with pytest.raises(ValueError, match="must be a duration"):
        a.Sample(1 * u.m)
    with pytest.raises(ValueError, match="must be positive"):
        a.Sample(0 * u.s)


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------
def test_and_makes_an_attrand():
    combined = a.Instrument("AIA") & a.Source("SDO")
    assert isinstance(combined, AttrAnd)
    assert len(combined) == 2


def test_and_is_associative():
    combined = a.Instrument("AIA") & a.Source("SDO") & a.Level(1.5)
    assert len(combined) == 3
    assert len(list(combined)) == 3


def test_or_makes_an_attror():
    combined = a.Instrument("AIA") | a.Instrument("HMI")
    assert isinstance(combined, AttrOr)
    assert len(combined) == 2


def test_or_of_identical_attrs_collapses():
    assert a.Instrument("AIA") | a.Instrument("AIA") == a.Instrument("AIA")


def test_and_distributes_over_or():
    combined = a.Time("2013-10-28", "2013-10-29") & (
        a.Instrument("AIA") | a.Instrument("HMI")
    )
    assert isinstance(combined, AttrOr)
    assert len(to_sum_of_products(combined)) == 2


def test_or_distributes_from_the_left():
    combined = (a.Instrument("AIA") | a.Instrument("HMI")) & a.Time(
        "2013-10-28", "2013-10-29"
    )
    assert len(to_sum_of_products(combined)) == 2


def test_or_of_ors():
    combined = (a.Instrument("AIA") | a.Instrument("HMI")) | a.Instrument("LASCO")
    assert len(combined) == 3


def test_and_with_a_non_attr():
    assert a.Instrument("AIA").__and__(3) is NotImplemented
    assert a.Instrument("AIA").__or__(3) is NotImplemented


def test_equality_and_hashing():
    assert a.Instrument("AIA") == a.Instrument("AIA")
    assert a.Instrument("AIA") != a.Instrument("HMI")
    assert len({a.Instrument("AIA"), a.Instrument("AIA")}) == 1


def test_collides():
    assert a.Instrument("AIA").collides(a.Instrument("HMI"))
    assert not a.Instrument("AIA").collides(a.Source("SDO"))


def test_and_helper():
    combined = and_(a.Instrument("AIA"), a.Source("SDO"), a.Level(1.5))
    assert len(to_sum_of_products(combined)[0]) == 3


def test_or_helper():
    combined = or_(a.Instrument("AIA"), a.Instrument("HMI"))
    assert len(to_sum_of_products(combined)) == 2


def test_helpers_need_arguments():
    with pytest.raises(ValueError, match="at least one"):
        and_()
    with pytest.raises(ValueError, match="at least one"):
        or_()


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------
def test_single_attr_flattens_to_one_term():
    assert to_sum_of_products(a.Instrument("AIA")) == [[a.Instrument("AIA")]]


def test_flattening_keeps_every_condition():
    query = a.Time("2013-10-28", "2013-10-29") & a.Instrument("AIA") & a.Level(1.5)
    terms = to_sum_of_products(query)
    assert len(terms) == 1
    assert len(terms[0]) == 3


def test_flattening_a_nested_or_inside_an_and():
    inner = AttrAnd([a.Instrument("AIA"), AttrOr([a.Level(1.5), a.Level(1.0)])])
    with pytest.raises(ValueError, match="could not be flattened"):
        to_sum_of_products(inner)


def test_repr_of_composites():
    assert "AttrAnd" in repr(a.Instrument("AIA") & a.Source("SDO"))
    assert "AttrOr" in repr(a.Instrument("AIA") | a.Instrument("HMI"))
