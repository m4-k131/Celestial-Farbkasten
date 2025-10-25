import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


from src.query_observation_lists import main as query_main

@pytest.fixture
def mock_astroquery():
    """Mocks the astroquery.mast.Observations class."""
    mock_obs_list = MagicMock()
    dummy_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    mock_obs_list.to_pandas.return_value = dummy_df
    with patch('src.query_observation_lists.Observations') as MockObs:
        MockObs.login.return_value = True
        MockObs.query_criteria.return_value = mock_obs_list
        yield MockObs, mock_obs_list

def test_main_fuzzy_search_calls_correctly(mock_astroquery, tmp_path):
    """
    Tests that main() correctly formats the target_name for a fuzzy search
    and saves the file to the correct path.
    """
    MockObs, mock_obs_list = mock_astroquery
    test_outdir = str(tmp_path)
    target = "NGC-628"
    with patch('src.query_observation_lists.os.getenv', return_value="fake-token"), \
         patch('src.query_observation_lists.os.makedirs') as mock_mkdir, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv:
        query_main(
            target_name=target,
            exact_name=False,
            outdir=test_outdir
        )
        # --- Assertions ---
        MockObs.login.assert_called_with("fake-token")
        MockObs.query_criteria.assert_called_with(
            target_name=f"*{target}*",  # Check fuzzy search
            dataproduct_type="image",
            calib_level=3,
            project="JWST"
        )
        mock_obs_list.to_pandas.assert_called_once()
        mock_mkdir.assert_called_with(test_outdir, exist_ok=True)
        expected_filename = os.path.join(test_outdir, f"{target}.csv")
        mock_to_csv.assert_called_with(expected_filename, index=False)
