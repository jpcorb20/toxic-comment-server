import json
from pytest import fixture
from mock import patch
from main import app


@fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@patch("main.ml_infer_labels")
@patch("main.dr_infer_labels")
def test_main_route(mock_dr, mock_ml, client):
    mock_ml.return_value = {"toxic": 1}
    mock_dr.return_value = {"toxic": 1}

    res = client.get("/toxic_comment?text='You are dumb, but it is ok.'")
    assert res.status_code == 200
    assert json.loads(res.data.decode("utf8"))["toxic"] == 1

    res = client.get("/toxic_comment?text='You are dumb, but it is ok.'&model=distilroberta")
    assert res.status_code == 200
    assert json.loads(res.data.decode("utf8"))["toxic"] == 1

    res = client.get("/toxic_comment")
    assert res.status_code == 400
