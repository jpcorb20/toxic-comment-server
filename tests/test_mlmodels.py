import os
import shutil
from mlmodels import load_models, infer_labels
from mock import patch, mock_open, MagicMock


@patch("mlmodels.load_models")
def test_infer_labels(mock_load):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]

    with patch.dict("mlmodels.models", {"toxic": mock_model}):
        assert infer_labels("")["toxic"] == 1

    with patch.dict("mlmodels.models", {}):
        assert infer_labels("") == {}


@patch("mlmodels.pickle")
def test_load_models(mock_pickle):
    if not os.path.exists("models/test"):
        os.mkdir("models/test")

    with open("models/test/model.pickle", "wb") as fp:
        fp.write(b"test")

    with patch('__main__.open', mock_open(), create=False):
        assert load_models("test") is None

    shutil.rmtree("models/test")
