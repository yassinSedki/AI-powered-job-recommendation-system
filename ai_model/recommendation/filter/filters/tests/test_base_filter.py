import pytest
import pandas as pd

from ai_model.recommendation.filter.filters.base_filter import (
    BaseJobFilter,
    FilterValidationError,
    FilterApplicationError,
)


def test_base_filter_is_abstract():
    # Attempting to instantiate BaseJobFilter should raise TypeError due to abstract methods
    with pytest.raises(TypeError):
        BaseJobFilter("Abstract")


class DummyFilter(BaseJobFilter):
    def __init__(self):
        super().__init__("Dummy")
        self.filter_column = "dummy_col"

    def apply_filter(self, df: pd.DataFrame, filter_criteria, **kwargs) -> pd.DataFrame:
        self.validate_dataframe(df, [self.filter_column])
        if not self.validate_criteria(filter_criteria):
            raise FilterValidationError("Invalid criteria")
        # Simple passthrough: return rows where dummy_col matches any criteria
        criteria = filter_criteria if isinstance(filter_criteria, list) else [filter_criteria]
        return df[df[self.filter_column].isin(criteria)]

    def validate_criteria(self, filter_criteria) -> bool:
        # Accept non-empty string or list of non-empty strings
        if isinstance(filter_criteria, str):
            return bool(filter_criteria.strip())
        if isinstance(filter_criteria, list):
            return all(isinstance(c, str) and c.strip() for c in filter_criteria)
        return False

    def get_available_options(self, df: pd.DataFrame):
        self.validate_dataframe(df, [self.filter_column])
        return sorted([v for v in df[self.filter_column].dropna().unique().tolist() if str(v).strip()])


def make_sample_df():
    return pd.DataFrame({
        'dummy_col': ['A', 'B', 'C', '', None, 'A'],
        'Job_Id': [1, 2, 3, 4, 5, 6]
    })


def test_validate_dataframe_empty_raises_value_error():
    f = DummyFilter()
    with pytest.raises(ValueError):
        f.validate_dataframe(pd.DataFrame(), [f.filter_column])


def test_validate_dataframe_none_raises_value_error():
    f = DummyFilter()
    with pytest.raises(ValueError):
        f.validate_dataframe(None, [f.filter_column])


def test_validate_dataframe_missing_columns_raises_key_error():
    f = DummyFilter()
    bad_df = pd.DataFrame({'wrong': [1, 2]})
    with pytest.raises(KeyError):
        f.validate_dataframe(bad_df, [f.filter_column])


def test_get_filter_summary_includes_expected_keys():
    f = DummyFilter()
    df = make_sample_df()
    summary = f.get_filter_summary(df)
    assert summary["filter_name"] == "Dummy"
    assert summary["total_jobs"] == len(df)
    assert isinstance(summary["available_options"], list)
    assert summary["filter_column"] == f.filter_column


def test_apply_filter_and_convenience_flow():
    f = DummyFilter()
    df = make_sample_df()
    out = f.apply_filter(df, 'A')
    assert set(out['dummy_col']) == {'A'}


def test_custom_exceptions_can_be_raised_and_caught():
    # FilterValidationError
    with pytest.raises(FilterValidationError):
        raise FilterValidationError("bad input")
    # FilterApplicationError
    with pytest.raises(FilterApplicationError):
        raise FilterApplicationError("failed apply")