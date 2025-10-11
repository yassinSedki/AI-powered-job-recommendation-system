import pytest
import pandas as pd

from ai_model.recommendation.filter.filters.work_type_filter import (
    WorkTypeFilter,
    FilterValidationError,
    filter_by_work_type,
)
from ai_model.recommendation.filter.filters.base_filter import FilterApplicationError


def make_sample_df():
    return pd.DataFrame({
        'Work Type': ['Contract', 'Part-time', 'Internship', 'Full-time', 'Contract', None, ''],
        'Job_Id': [1, 2, 3, 4, 5, 6, 7]
    })


def test_apply_filter_exact_match_single():
    df = make_sample_df()
    f = WorkTypeFilter()
    out = f.apply_filter(df, 'Contract')
    assert set(out['Work Type']) == {'Contract'}


def test_apply_filter_multiple_types():
    df = make_sample_df()
    f = WorkTypeFilter()
    out = f.apply_filter(df, ['Part-time', 'Internship'])
    assert set(out['Work Type']) == {'Part-time', 'Internship'}


def test_invalid_criteria_raises():
    df = make_sample_df()
    f = WorkTypeFilter()
    with pytest.raises(FilterValidationError):
        f.apply_filter(df, 'Remote')


def test_missing_column_raises_application_error():
    df = pd.DataFrame({'wrong_column': ['Contract', 'Part-time']})
    f = WorkTypeFilter()
    with pytest.raises(FilterApplicationError):
        f.apply_filter(df, 'Contract')


def test_get_available_options_returns_unique_sorted():
    df = make_sample_df()
    f = WorkTypeFilter()
    opts = f.get_available_options(df)
    assert isinstance(opts, list)
    assert 'Contract' in opts and 'Part-time' in opts


def test_convenience_function_filter_by_work_type():
    df = make_sample_df()
    out = filter_by_work_type(df, 'Full-time')
    assert set(out['Work Type']) == {'Full-time'}


def test_apply_filter_each_supported_type_and_print_results(capsys):
    df = make_sample_df()
    f = WorkTypeFilter()
    # Iterate over each supported work type and print counts and rows
    for wt in WorkTypeFilter.SUPPORTED_WORK_TYPES:
        out = f.apply_filter(df, wt)
        print(f"WorkType='{wt}' -> count={len(out)} | Job_Ids={out['Job_Id'].tolist()} | Values={out['Work Type'].tolist()}")
    captured = capsys.readouterr()
    # Ensure our diagnostic output contains at least one line for each work type
    for wt in WorkTypeFilter.SUPPORTED_WORK_TYPES:
        assert f"WorkType='{wt}'" in captured.out



def test_apply_filter_with_all_categories_list_shows_total(capsys):
    df = make_sample_df()
    f = WorkTypeFilter()
    out = f.apply_filter(df, WorkTypeFilter.SUPPORTED_WORK_TYPES)
    print(f"All categories -> count={len(out)} | Job_Ids={out['Job_Id'].tolist()} | Values={out['Work Type'].tolist()}")
    captured = capsys.readouterr()
    assert "All categories" in captured.out
    # In our sample, Contract appears twice, others once -> total 5
    assert len(out) == 5



def test_case_sensitivity_exploration_prints_results(capsys):
    # Build a DataFrame with lowercased and trimmed variations to demonstrate exact, case-sensitive matching
    df = pd.DataFrame({
        'Work Type': ['contract', ' part-time ', 'INTERNSHIP', 'Full-time'],
        'Job_Id': [101, 102, 103, 104]
    })
    f = WorkTypeFilter()

    # Try filtering using the canonical supported values
    for wt in WorkTypeFilter.SUPPORTED_WORK_TYPES:
        out = f.apply_filter(df, wt)
        print(f"CaseSensitivity: filter='{wt}' -> count={len(out)} | Job_Ids={out['Job_Id'].tolist()} | Values={out['Work Type'].tolist()}")

    captured = capsys.readouterr()
    # Only 'Full-time' should match exactly in this DataFrame
    assert "filter='Full-time'" in captured.out



def test_resolves_column_alias_work_type():
    # Use 'Work_Type' alias to confirm alias resolution works
    df = pd.DataFrame({
        'Work_Type': ['Contract', 'Part-time', 'Internship', 'Full-time'],
        'Job_Id': [201, 202, 203, 204]
    })
    f = WorkTypeFilter()
    out = f.apply_filter(df, 'Contract')
    assert len(out) == 1
    assert set(out['Work_Type']) == {'Contract'}