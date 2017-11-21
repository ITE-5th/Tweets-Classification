from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from pyarabic.araby import strip_tashkeel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from transformers.length_transformer import LengthTransformer
from transformers.marks_count_transformer import MarksCountTransformer
from transformers.sentences_count_transformer import SentencesCountTransformer
from transformers.words_count_transformer import WordsCountTransformer


def pre_process(data):
    data = data.dropna(how="any")
    data.loc[:, "sentiment"] = data.loc[:, "sentiment"].apply(lambda x: int(x.lower().strip() == "yes"))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: x.strip())
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: strip_tashkeel(x))
    # TODO: are we sure that we should delete all the tweets with any english letter?
    data = data.loc[~data.loc[:, "tweet"].str.contains("[a-zA-Z]"), :]
    data = data.drop_duplicates(subset="tweet")
    return data


if __name__ == '__main__':
    with open("Data/arabic_stop_words.txt", "r", encoding="utf-8") as file:
        arabic_stop_words = file.readlines()
    file_path = "Data/tweet_data_v2.txt"
    tweets = pd.read_csv(file_path, sep=r"\s?\|\|\s?", skip_blank_lines=True, engine='python', encoding="utf-8")
    tweets = pre_process(tweets)
    X, y = tweets["tweet"].astype(np.str), tweets["sentiment"].astype(np.str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rf_parameters = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_features': ['log2', 'sqrt', 0.8]
    }

    svc_parameters = {
        'svc__C': [10, 100, 1000],
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf', "linear"]
    }

    nb_parameters = {
        'nb__alpha': (1, 0.1, 0.01, 0.001, 0.00001)
    }

    # change to svc or nb if you want, svc will take a long time
    method = "rf"

    if method == "rf":
        clz = ("rf", RandomForestClassifier())
        parameters = rf_parameters
    elif method == "svc":
        clz = ("svc", SVC())
        parameters = svc_parameters
    else:
        clz = ("nb", BernoulliNB())
        parameters = nb_parameters

    pipeline = Pipeline([
        ('features_extraction', FeatureUnion([
            ('tfidf', TfidfVectorizer(stop_words=set(arabic_stop_words), norm="l2")),
            ('tweet_length', LengthTransformer()),
            ('marks_count', MarksCountTransformer()),
            ('sentences_count', SentencesCountTransformer()),
            ('words_count', WordsCountTransformer()),
            # ('deny_words', WordsCountTransformer()),
        ])),
        clz
    ])

    grid = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=10, n_jobs=cpu_count())
    grid.fit(X_train, y_train)
    estimator = grid.best_estimator_
    predicted = estimator.predict(X_test)
    error = log_loss(y_test, predicted)
    print("cross entropy error = {}".format(error))
    acc = np.sum(predicted == y_test) / y_test.shape[0] * 100
    print("accuracy = {}".format(acc))
