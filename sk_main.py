import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics.classification import log_loss
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from transformers.length_transformer import LengthTransformer
from multiprocessing import cpu_count
from transformers.marks_count_transformer import MarksCountTransformer
from transformers.sentences_count_transformer import SentencesCountTransformer
from transformers.words_count_transformer import WordsCountTransformer


def preprocess(tweets):
    tweets["sentiment"] = tweets["sentiment"].apply(lambda x: int(x.strip() == "yes"))
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.strip())
    return tweets[~tweets["tweet"].str.contains("http")]


with open("arabic_stop_words.txt", "r") as file:
    arabic_stop_words = file.readline()
file_path = "Driving_Data_Cleaned_with_hashtag.txt"
tweets = pd.read_csv(file_path, sep=r"\s?\|\|\s?", skip_blank_lines=True)
tweets = preprocess(tweets)
X, y = tweets["tweet"].astype(np.str), tweets["sentiment"].astype(np.str)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipeline = Pipeline([
    ('features_extraction', FeatureUnion([
        ('tfidf', TfidfVectorizer(stop_words=set(arabic_stop_words), norm="l2")),
        ('tweet_length', LengthTransformer()),
        ('marks_count', MarksCountTransformer()),
        ('sentences_count', SentencesCountTransformer()),
        ('words_count', WordsCountTransformer()),
    ])),
    # ('rf', RandomForestClassifier())
    ("svc", SVC())
    # ("nb", BernoulliNB())
])

tree_parameters = {
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

# grid = GridSearchCV(estimator=pipeline, param_grid=tree_parameters, cv=10, n_jobs=cpu_count())
grid = GridSearchCV(estimator=pipeline, param_grid=svc_parameters, cv=10, n_jobs=cpu_count())
# grid = GridSearchCV(estimator=pipeline, param_grid=nb_parameters, cv=10, n_jobs=cpu_count())
grid.fit(X_train, y_train)
estimator = grid.best_estimator_
predicted = estimator.predict(X_test)
# error = log_loss(y_test, predicted)
# print("cross entropy error = {}".format(error))
acc = np.sum(predicted == y_test) / predicted.shape[0] * 100
print("accuracy = {}".format(acc))