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

CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


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

train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

X_train = train.comment_text
X_test = test.comment_text
print(X_train.shape)
print(X_test.shape)

# Define a pipeline combining a text feature extractor with multi lable classifier

SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])

xg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(XGBClassifier())),
])

print("SVC")

for category in CATEGORIES:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(f1_score(test[category], prediction)))

print()
print("Logistic")

for category in CATEGORIES:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(f1_score(test[category], prediction)))

print()
print("XGBoost")

model_dict = dict()
for category in CATEGORIES:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    xg_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = xg_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(f1_score(test[category], prediction)))
    model_dict[category] = deepcopy(xg_pipeline)

with open("models/xgboost/xgboost_model.pickle", "wb") as fp:
    pickle.dump(model_dict, fp)

print()

# XGBoost results :
# ... Processing toxic
# Test accuracy is 0.7084481725584183
# ... Processing severe_toxic
# Test accuracy is 0.2849462365591398
# ... Processing obscene
# Test accuracy is 0.7864593583940169
# ... Processing threat
# Test accuracy is 0.32710280373831774
# ... Processing insult
# Test accuracy is 0.6481149012567325
# ... Processing identity_hate
# Test accuracy is 0.36036036036036034
