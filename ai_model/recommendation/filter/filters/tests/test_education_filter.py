import pytest
import pandas as pd

from ai_model.recommendation.filter.filters.education_filter import (
    EducationFilter,
    FilterValidationError,
    FilterApplicationError,
    filter_by_education,
)


def make_sample_df():
    return pd.DataFrame({
        'qualification_category': [
            'Bachelor', 'Master', 'PhD', 'bachelor', 'master', None, ''
        ],
        'Job_Id': [1, 2, 3, 4, 5, 6, 7]
    })


def test_apply_filter_single_includes_case_insensitive_matches():
    df = make_sample_df()
    f = EducationFilter()
    out = f.apply_filter(df, 'Bachelor')
    # Should include both 'Bachelor' and 'bachelor' after .str.title() normalization
    assert set(out['qualification_category'].str.title()) == {'Bachelor'}


def test_apply_filter_multiple_levels_raises():
    df = make_sample_df()
    f = EducationFilter()
    # Passing a list should raise FilterValidationError now that only single value is allowed
    with pytest.raises(FilterValidationError):
        f.apply_filter(df, ['Bachelor', 'PhD'])


def test_invalid_criteria_raises():
    df = make_sample_df()
    f = EducationFilter()
    with pytest.raises(FilterValidationError):
        f.apply_filter(df, 'Diploma')


def test_missing_column_raises_application_error():
    df = pd.DataFrame({'wrong_column': ['Bachelor', 'Master']})
    f = EducationFilter()
    # Base validate_dataframe raises KeyError, but EducationFilter wraps unexpected exceptions
    # into FilterApplicationError, so we expect FilterApplicationError here.
    with pytest.raises(FilterApplicationError):
        f.apply_filter(df, 'Bachelor')


def test_get_available_options_returns_sorted_unique():
    df = make_sample_df()
    f = EducationFilter()
    opts = f.get_available_options(df)
    # Should include the exact unique values present (excluding None), sorted by the implementation
    # Note: get_available_options does not normalize case; it returns dataset's unique values
    expected_values = {'Bachelor', 'Master', 'PhD', 'bachelor', 'master', ''}
    assert set(opts) == expected_values


def test_convenience_function_filter_by_education():
    df = make_sample_df()
    out = filter_by_education(df, 'Master')
    assert set(out['qualification_category'].str.title()) == {'Master'}


def test_case_sensitive_filtering_excludes_lowercase():
    df = make_sample_df()
    f = EducationFilter()
    out = f.apply_filter(df, 'Bachelor', case_sensitive=True)
    # Should not include 'bachelor' when case_sensitive=True
    assert set(out['qualification_category']) == {'Bachelor'}