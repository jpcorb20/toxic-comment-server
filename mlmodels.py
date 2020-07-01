import pickle

models = None


def load_models(model_name="svc"):
    """
    Load dict in global variable with models OneVsRest.
    :return: None.
    """
    global models
    with open("models/%s/model.pickle" % model_name, "rb") as fp:
        models = pickle.load(fp)


def infer_labels(text):
    """
    Predict class dict with key as toxic classes and value in {0,1}.
    :param text: comment (str).
    :return: dict.
    """
    global models

    if models is None:
        load_models()

    return {k: int(m.predict([text])[0]) for k, m in models.items()}
