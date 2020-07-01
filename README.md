# Toxic Comment Classifier Server

Models to detect hateful comments served with flask trained on the well-known Kaggle's dataset.

## Dataset

Kaggle's [Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
is a multi-label classification task containing a set of 6 possible labels:

- *toxic*
- *severe_toxic*
- *obscene*
- *threat*
- *insult*
- *identity_hate*

Thus, any sample can have between 0 and 6 labels at a time.

## References

Since it is a very known task in scientific and gray literature, we found interesting previous work on _Toward Data Science_:

- ML and NLP pipeline approach: [Source 1](https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5).

- Transformer-based approach: [Source 2](https://towardsdatascience.com/transformers-for-multilabel-classification-71a1a0daf5e1).

Both projects have very bad coding.

## Analysis

This part correspond to this script:

    python analysis.py

The dataset is highly imbalanced with near 90% of the 159,571 samples not labeled. The distribution of labels is:

- *toxic* : 15,294 comments
- *obscene*: 8449 comments
- *insult*: 7,877 comments
- *severe_toxic*: 1,595 comments
- *identity_hate*: 1,405 comments
- *threat*: 478 comments

The length of comments are skewed distributions:

- characters are on average (394 +/- 590).
- average length of ( 80 +/- 120 ) tokens.

we also noted a lot of noise and language contractions in text since they are comments. We cleaned them in the NLP pipeline.

## Models

We have two approaches: the ML pipeline and the transformer-based one.
The first contain baseline classification models with TF-IDF features in one vs rest setup.
The second consist of current SOTA model of which we cannot simply ignore even if they are harder to put into production.

We do not focus on hyperparameter optimization in this project.

### Classical ML NLP pipeline

To train all baseline models run:

    python baselines.py

#### Linear SVC

Model based on SVM architecture for classification known in NLP to be SOTA before the deep learning era.

F1 scores for svc:
- toxic: 0.768866
- severe_toxic: 0.404000
- obscene: 0.787289
- threat: 0.314815
- insult: 0.670479
- identity_hate: 0.378378

Macro-F1: 0.553971

#### Logistic Regression

Model well-known in ML with good interpretability. It is often too simple for task.

F1 scores for logistic:
- toxic: 0.721715
- severe_toxic: 0.376068
- obscene: 0.733263
- threat: 0.193548
- insult: 0.622429
- identity_hate: 0.238636

Macro-F1: 0.480943

#### XGBoost

An ensemble classifier that is considered very effective in many task by the ML community.

F1 scores for xgboost:
- toxic: 0.710846
- severe_toxic: 0.301969
- obscene: 0.786611
- threat: 0.350877
- insult: 0.648352
- identity_hate: 0.387255

Macro-F1: 0.530985

### Pre-trained Transformer

#### DistilRoBERTa

## Install dependencies

This project uses the _pip_ command to install dependencies without issues.

    pip install -r requirements.txt

For Windows user, the _torch_ dependency require this specific installation:

    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

## Tests

To run the tests, there is this command:

    pytest ...
