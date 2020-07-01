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

# Inspired by Susan Li's code:
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

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
    (r"\'scuse", " excuse "),
    ('\W', ' '),
    ('\s+', ' ')
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
    print("F1 scores for %s" % name)

    model_dict = dict()
    for category in TOXIC_CATEGORIES:
        pipeline.fit(train.comment_text, train[category])

        prediction = pipeline.predict(test.comment_text)
        print('%20s: %.6f' % (category, f1_score(test[category], prediction)))

        model_dict[category] = deepcopy(xg_pipeline)

    if save:
        with open("models/%s/model.pickle" % name, "wb") as fp:
            pickle.dump(model_dict, fp)


# Defining three pipelines combining a TFIDF extractor and multilabel classifiers.

SVC_pipeline = Pipeline([
                 ('tfidf', TfidfVectorizer(stop_words=STOPWORDS)),
                 ('clf', OneVsRestClassifier(LinearSVC(random_state=RDN_NUM), n_jobs=1)),
              ])

LogReg_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=STOPWORDS)),
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', random_state=RDN_NUM), n_jobs=1)),
                  ])

xg_pipeline = Pipeline([
                  ('tfidf', TfidfVectorizer(stop_words=STOPWORDS)),
                  ('clf', OneVsRestClassifier(XGBClassifier(random_state=RDN_NUM))),
              ])

pipelines = {
    "svc": SVC_pipeline,
    "logistic": LogReg_pipeline,
    "xgboost": xg_pipeline
}

if __name__ == "__main__":

    df = pd.read_csv("data/train.csv")
    print(df.shape)
    df['comment_text'] = df['comment_text'].map(lambda com: clean_text(com))

    train, dev = train_test_split(df, random_state=RDN_NUM, test_size=0.33, shuffle=True)

    for name, pipeline in pipelines.items():
        run_pipeline(train, dev, name, pipeline)
        print()
