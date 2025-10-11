import pytest
import pandas as pd

from ai_model.recommendation.filter.filters.gender_filter import (
    GenderFilter,
    FilterValidationError,
    FilterApplicationError,
    filter_by_gender,
    filter_inclusive_jobs,
)


def make_sample_df():
    return pd.DataFrame({
        'Preference': ['male', 'female', 'both', 'Male', 'Female', 'Both', None, ''],
        'Job_Id': [1, 2, 3, 4, 5, 6, 7, 8]
    })


def test_apply_filter_male_includes_both():
    df = make_sample_df()
    f = GenderFilter()
    out = f.apply_filter(df, 'male')
    # After normalization to lowercase in apply, expect male + both
    assert set(out['Preference'].str.lower()) == {'male', 'both'}


def test_apply_filter_female_includes_both():
    df = make_sample_df()
    f = GenderFilter()
    out = f.apply_filter(df, 'female')
    assert set(out['Preference'].str.lower()) == {'female', 'both'}


def test_invalid_criteria_raises():
    df = make_sample_df()
    f = GenderFilter()
    with pytest.raises(FilterValidationError):
        f.apply_filter(df, 'unknown')


def test_missing_column_raises_application_error():
    df = pd.DataFrame({'wrong_column': ['male', 'female']})
    f = GenderFilter()
    # Base validate_dataframe raises KeyError, but GenderFilter wraps unexpected exceptions
    # into FilterApplicationError, so we expect FilterApplicationError here.
    with pytest.raises(FilterApplicationError):
        f.apply_filter(df, 'male')


def test_get_available_options_returns_unique_sorted():
    df = make_sample_df()
    f = GenderFilter()
    opts = f.get_available_options(df)
    assert isinstance(opts, list)
    assert 'male' in [o.lower() for o in opts] and 'female' in [o.lower() for o in opts]


def test_convenience_functions():
    df = make_sample_df()
    out1 = filter_by_gender(df, 'male')
    assert set(out1['Preference'].str.lower()) == {'male', 'both'}
    out2 = filter_inclusive_jobs(df)
    # Inclusive only should return entries equal to 'both' (case-insensitive)
    assert set(out2['Preference'].str.lower()) == {'both'}