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

Both projects have very bad coding, mistakes and wrong evaluation metrics, but we inspired our approaches from them to start from some points and push the server into production.

We want to constrast baseline approach like Source 1 versus SOTA one like Source 2.

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
The second consist of current SOTA model which have revolutionized NLP even if they are harder to put into production.

We do not focus on hyperparameter optimization in this project. We will take default know configuration and use our multiple models to make a final choice.

We evaluate our models with the Macro F1 metric which the average of individual F1 scores.
We use F1 to optimize both precise detection and finding most toxic comments (recall).

### Classical ML NLP pipeline

To train all baseline models run:

    python baselines.py

##### Linear SVC

Model based on SVM architecture for classification known in NLP to be SOTA before the deep learning era.

F1 scores for svc:
- toxic: 0.768866
- severe_toxic: 0.404000
- obscene: 0.787289
- threat: 0.314815
- insult: 0.670479
- identity_hate: 0.378378

Macro-F1: 0.553971

##### Logistic Regression

Model well-known in ML with good interpretability. It is equivalent to one artificial neuron but often too simple for high-level task.

F1 scores for logistic:
- toxic: 0.721715
- severe_toxic: 0.376068
- obscene: 0.733263
- threat: 0.193548
- insult: 0.622429
- identity_hate: 0.238636

Macro-F1: 0.480943

##### XGBoost

An ensemble classifier using boosting that is considered very effective in many tasks by the ML community.

F1 scores for xgboost:
- toxic: 0.710846
- severe_toxic: 0.301969
- obscene: 0.786611
- threat: 0.350877
- insult: 0.648352
- identity_hate: 0.387255

Macro-F1: 0.530985

### Pre-trained Transformer

In the last couple of years, pre-trained transformer models have completely change the NLP field and SOTA.
They are very huge models pre-trained on massive corpora.
Furthermore, they have established performances often above human levels (e.g.: SQuaD) since they are very effective for high-level task like detecting toxic comments.

To fine-tune from pre-trained model run:

    python transformer_finetune.py

The final fine-tune model was pushed on huggingface.co's S3 bucket as:

    jpcorb20/toxic-detector-distilroberta

It is directly downloaded and used by the _distilroberta_ module.

##### DistilRoBERTa

We chose the DistilRoBERTa model for two reasons: RoBERTa is the most optimized and robust available model
and the distillation process makes it smaller by 30-40% while maintaining most performances
which is great for production.

F1 scores for distilroberta:
- toxic: 0.72
- severe_toxic: 0.38
- obscene: 0.72
- threat: 0.52
- insult: 0.69
- identity_hate: 0.60

Macro-F1: 0.61

Thus, this model is the most performing one with great improvements on most labels with best improvements on _threat_ and _identity_hate_.

## Install dependencies

This project uses the _pip_ command to install dependencies without issues.

    pip install -r requirements.txt

For Windows user, the _torch_ dependency require this specific installation:

    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

## Tests

To run the tests, there is this command:

    pytest --cov=. --cov-report=term-missing tests\

## Production-ready Docker version

All tests should be completed before building and releasing the container.

Build the image for better reproducibility and get a production-ready image (might take some time):

    docker build -t toxic-comment-server

Run a container with environment variables:

    docker run -p 8080:8080 --env DEBUG=0 --env [...OTHER ENV...] toxic-comment-server

## Futher works

- Finish all docstring.
- 100% coverage of code with unit tests (mostly transformer code).
- Enhance security with POST request instead of GET in main.py.
- Do hyperparameter optimization to enhance model performances.
- Generate a TinyBert version of transformer for very optimized production-ready model.
- Add a CI for better upgrades and smooth deployments.
- Deploy container.
