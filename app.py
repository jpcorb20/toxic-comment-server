import os
import pickle
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv
load_dotenv()

debug = bool(int(int(os.environ["DEBUG"])))

app = Flask(__name__)

models = None


def load_models():
    """
    Load dict in global variable with models OneVsRest.
    :return: None.
    """
    global models

    with open("models/xgboost/xgboost_model.pickle", "rb") as fp:
        models = pickle.load(fp)


def predict(text):
    """
    Predict class dict with key as toxic classes and value in {0,1}.
    :param text: comment (str).
    :return: dict.
    """
    global models

    if models is None:
        load_models()

    return {k: int(m.predict([text])[0]) for k, m in models.items()}


@app.route("/toxic_comment", methods=["GET"])
def main_route():
    """
    Route that predict whether a comment (a string in text GET param) is toxic among 6 classes in multi-label setting.
    :return: JSON object in which keys are the 6 classes with {0,1} as values.
    """
    if "text" in request.args:
        text = request.args["text"]
        labels = predict(text)
        return jsonify(labels)
    else:
        return abort(400)


if __name__ == "__main__":
    load_models()
    app.run(debug=debug,
            host=os.environ["FLASK_HOST"],
            port=int(os.environ["FLASK_PORT"]))
