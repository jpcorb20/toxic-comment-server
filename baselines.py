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
stop_words = set(stopwords.words('english'))

# Inspired by Susan Li's code:
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

TOXIC_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

RDN_NUM = 42


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


df = pd.read_csv("data/train.csv")
df['comment_text'] = df['comment_text'].map(lambda com: clean_text(com))

train, test = train_test_split(df, random_state=RDN_NUM, test_size=0.33, shuffle=True)

# Define a pipeline combining a text feature extractor with multi label classifier

SVC_pipeline = Pipeline([
                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                 ('clf', OneVsRestClassifier(LinearSVC(random_state=RDN_NUM), n_jobs=1)),
              ])

LogReg_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', random_state=RDN_NUM), n_jobs=1)),
                  ])

xg_pipeline = Pipeline([
                  ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                  ('clf', OneVsRestClassifier(XGBClassifier(random_state=RDN_NUM))),
              ])

pipelines = {
    "svc": SVC_pipeline,
    "logistic": LogReg_pipeline,
    "xgboost": xg_pipeline
}


def run_pipeline(train, test, name, pipeline, save=True):
    print(name)

    model_dict = dict()
    for category in TOXIC_CATEGORIES:
        pipeline.fit(train.comment_text, train[category])

        prediction = pipeline.predict(test.comment_text)
        print('F1 score of %20s: %.6f' % (category, f1_score(test[category], prediction)))

        model_dict[category] = deepcopy(xg_pipeline)

    if save:
        with open("models/%s/model.pickle" % name, "wb") as fp:
            pickle.dump(model_dict, fp)


if __name__ == "__main__":
    for name, pipeline in pipelines.items():
        run_pipeline(train, test, name, pipeline)
        print()
