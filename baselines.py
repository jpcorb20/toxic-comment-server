import re
import pickle
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from nltk.corpus import stopwords

RDN_NUM = 42

TOXIC_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

STOPWORDS = set(stopwords.words('english'))

CLEAN_TEXT_ELEMENTS = [
    (r"what's", "what is "),
    (r"\'s", " "),
    (r"\'ve", " have "),
    (r"can't", "can not "),
    (r"n't", " not "),
    (r"i'm", "i am "),
    (r"\'re", " are "),
    (r"\'d", " would "),
    (r"\'ll", " will "),
    (r'\W', ' '),
    (r'\s+', ' ')
]


def clean_text(text):
    """
    Clean contractions in text.
    :param text: raw text (str).
    :return: cleaned text (str).
    """

    text = text.lower()

    for sub in CLEAN_TEXT_ELEMENTS:
        text = re.sub(*sub, text)

    text = text.strip(' ')

    return text


def mount_pipeline(classifier, classifier_params):
    """
    Mount classifier in OneVsRest configuration pipeline with TFIDF vectorizer as previous stage.
    :param classifier: sklearn classifier class.
    :param classifier_params: params of classifier.
    :return: mounted pipeline (sklearn.pipeline.Pipeline).
    """
    return Pipeline([
             ('tfidf', TfidfVectorizer(stop_words=STOPWORDS)),
             ('clf', OneVsRestClassifier(classifier(**classifier_params), n_jobs=1)),
          ])


def run_pipeline(train, test, name, pipeline, save=True):
    """
    Run a sklearn pipeline on train and test.
    :param train: train set (pandas.DataFrame).
    :param test: test set (pandas.DataFrame).
    :param name: name of model (str).
    :param pipeline: ML classification pipeline (sklearn.pipeline.Pipeline).
    :param save: save pickle in diretory models/<name>/model.pickle (boolean).
    :return: None.
    """
    print("F1 scores for %s:" % name)

    model_dict = dict()
    for category in TOXIC_CATEGORIES:
        pipeline.fit(train.comment_text, train[category])

        prediction = pipeline.predict(test.comment_text)
        print('%20s: %.6f' % (category, f1_score(test[category], prediction)))

        model_dict[category] = deepcopy(pipeline)

    if save:
        with open("models/%s/model.pickle" % name, "wb") as fp:
            pickle.dump(model_dict, fp)


# Defining three pipelines combining a TFIDF extractor and multilabel classifiers.

pipelines = {
    "svc": (LinearSVC, dict(random_state=RDN_NUM)),
    "logistic": (LogisticRegression, dict(solver='sag', random_state=RDN_NUM)),
    "xgboost": (XGBClassifier, dict(random_state=RDN_NUM))
}

pipelines = {k: mount_pipeline(*p) for k, p in pipelines.items()}

if __name__ == "__main__":

    df = pd.read_csv("data/train.csv")
    df['comment_text'] = df.comment_text.map(lambda t: clean_text(t))

    train, dev = train_test_split(df, random_state=RDN_NUM, test_size=0.20, shuffle=True)

    for name, pipeline in pipelines.items():
        run_pipeline(train, dev, name, pipeline)
        print()
