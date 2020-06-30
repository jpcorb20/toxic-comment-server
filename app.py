import os
from flask import Flask, abort
from dotenv import load_dotenv
load_dotenv()

debug = bool(int(int(os.environ["DEBUG"])))

app = Flask(__name__)


@app.route("/")
def main_route():
    return abort(404)


if __name__ == "__main__":
    app.run(debug=debug,
            host=os.environ["FLASK_HOST"],
            port=int(os.environ["FLASK_PORT"]))
