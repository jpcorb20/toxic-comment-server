import os
from flask import Flask, request, abort, jsonify
from flask_basicauth import BasicAuth
from dotenv import load_dotenv

from distilroberta import infer_labels as dr_infer_labels
from mlmodels import infer_labels as ml_infer_labels

load_dotenv()

DEBUG = bool(int(int(os.environ["DEBUG"])))

app = Flask(__name__)

if not DEBUG:
    # Authentification to protect access.
    app.config['BASIC_AUTH_USERNAME'] = os.environ['BASIC_AUTH_USERNAME']
    app.config['BASIC_AUTH_PASSWORD'] = os.environ['BASIC_AUTH_PASSWORD']
    app.config['BASIC_AUTH_FORCE'] = True
    basic_auth = BasicAuth(app)


@app.route("/toxic_comment", methods=["GET"])
def main_route():
    """
    Route that infer_labels whether a comment (a string in text GET param) is toxic among 6 classes in multi-label setting.
    :return: JSON object in which keys are the 6 classes with {0,1} as values.
    """
    if "text" in request.args:
        text = request.args["text"]

        labels = dict()
        if "model" in request.args:

            if request.args["model"] == "distilroberta":
                labels = dr_infer_labels(text)

        else:
            labels = ml_infer_labels(text)
        
        return jsonify(labels)
    else:
        return abort(400)


if __name__ == "__main__":
    app.run(debug=DEBUG,
            host=os.environ["FLASK_HOST"],
            port=int(os.environ["FLASK_PORT"]))
