from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class SentencesCountTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, delimiters=None):
        if delimiters is None:
            delimiters = ["!", "?", ",", "،"]
        self.delimiters = delimiters

    def transform(self, tweets):
        temp = np.apply_along_axis(self.extract, 1, tweets.values.reshape(tweets.size, -1)).reshape(tweets.size, -1).astype(np.int)
        return temp

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def extract(self, arr):
        elem = arr[0]
        return np.array(sum(elem.count(delim) for delim in self.delimiters))
