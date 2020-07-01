import os
import pickle

models = dict()


def load_models(model_name="svc"):
    """
    Load dict in global variable with models OneVsRest.
    :return: None.
    """
    global models

    assert "models/%s/model.pickle" % model_name not in os.listdir(), "You should run baseline to generate models."

    with open("models/%s/model.pickle" % model_name, "rb") as fp:
        models = pickle.load(fp)


def infer_labels(text):
    """
    Predict class dict with key as toxic classes and value in {0,1}.
    :param text: comment (str).
    :return: dict.
    """
    global models

    if isinstance(models, dict) and len(models) == 0:
        load_models()

    return {k: int(m.predict([text])[0]) for k, m in models.items()}
