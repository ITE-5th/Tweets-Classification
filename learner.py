import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from preprocessing.preprocessing import pre_process
from transformers.deny_transformer import DenyTransformer
from transformers.length_transformer import LengthTransformer
from transformers.marks_count_transformer import MarksCountTransformer
from transformers.sentences_count_transformer import SentencesCountTransformer
from transformers.words_count_transformer import WordsCountTransformer


class Learner:
    def __init__(self, data_root_path: str, data_file_name: str):
        self.root_path = data_root_path
        self.data_file_name = data_file_name

    def learn(self):
        with open("{}/arabic_stop_words.txt".format(self.root_path), "r", encoding="utf-8") as file:
            arabic_stop_words = file.readlines()
        file_path = "{}/{}".format(self.root_path, self.data_file_name)
        tweets = pd.read_csv(file_path, sep=r"\s?\|\|\s?", skip_blank_lines=True, engine='python', encoding="utf-8")
        tweets = pre_process(tweets)
        X, y = tweets["tweet"].astype(np.str), tweets["sentiment"].astype(np.int)
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

        lr_parameters = {
            "lr__C": np.logspace(-4, 4, 3),
            # "pca__n_components": [20, 40, 64]
        }

        # change to rf, lr, svc, nb, or gb if you want, svc will take a long time
        method = "rf"

        if method == "rf":
            clz = ("rf", RandomForestClassifier())
            parameters = rf_parameters
        elif method == "svc":
            clz = ("svc", SVC())
            parameters = svc_parameters
        elif method == "nb":
            clz = ("nb", BernoulliNB())
            parameters = nb_parameters
        elif method == "gb":
            clz = ("gb", GaussianNB())
            parameters = {}
        elif method == "lr":
            clz = ("lr", LogisticRegression())
            parameters = lr_parameters
        else:
            clz = ("mlp", MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30, 30), alpha=1e-5))
            parameters = {}

        pipeline = Pipeline([
            ('features_extraction', FeatureUnion([
                ('tfidf',
                 TfidfVectorizer(stop_words=set(arabic_stop_words), norm="l2",
                                 tokenizer=nltk.tokenize.wordpunct_tokenize,
                                 analyzer="word", ngram_range=(1, 2))),
                ('tweet_length', LengthTransformer()),
                ('marks_count', MarksCountTransformer()),
                ('sentences_count', SentencesCountTransformer()),
                ('words_count', WordsCountTransformer()),
                ('deny_words_count', DenyTransformer()),
            ])),
            # ("feature_selection", FeatureUnion([
            #     ("pca", KernelPCA(40)),
            #     ("select_k_best", SelectKBest(k=5))
            # ])),
            # ("normalizer", StandardScaler()),
            # ('nys', Nystroem(kernel='', n_components=100)),
            clz
        ])

        grid = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=10)
        grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        predicted = estimator.predict(X_test)
        # error = log_loss(y_test, predicted)
        # print("cross entropy error = {}".format(error))
        acc = np.sum(predicted == y_test) / y_test.shape[0] * 100
        print("accuracy = {}".format(acc))
        joblib.dump(estimator, "models/predictor.pkl")


if __name__ == '__main__':
    # file_name = "tweet_data_v2.txt"
    file_name = "Driving_Data_Cleaned_with_hashtag.txt"
    learner = Learner("data", file_name)
    learner.learn()
