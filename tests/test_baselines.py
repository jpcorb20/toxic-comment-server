import shutil
from baselines import clean_text, run_pipeline
from mock import MagicMock, patch, mock_open


def test_clean_text():
    assert clean_text("I'm GOING there\n.") == "i am going there"


@patch("baselines.pickle")
@patch("baselines.f1_score")
def test_run_pipeline(mock_f1, mock_pickle):
    mock_f1.return_value = 1

    with patch('__main__.open', mock_open(), create=False):
        assert run_pipeline(MagicMock(), MagicMock(), "test", MagicMock()) is None

    shutil.rmtree("models/test")
