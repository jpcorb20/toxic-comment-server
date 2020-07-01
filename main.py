import os
from flask import Flask, request, abort, jsonify
from dotenv import load_dotenv

import distilroberta as dr
import mlmodels as ml

load_dotenv()

DEBUG = bool(int(int(os.environ["DEBUG"])))

app = Flask(__name__)


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
                labels = dr.infer_labels(text)

        else:
            labels = ml.infer_labels(text)
        
        return jsonify(labels)
    else:
        return abort(400)


if __name__ == "__main__":
    app.run(debug=DEBUG,
            host=os.environ["FLASK_HOST"],
            port=int(os.environ["FLASK_PORT"]))
