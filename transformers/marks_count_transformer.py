from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class MarksCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, marks=None):
        if marks is None:
            marks = ["?", "!"]
        self.marks = marks

    def transform(self, tweets):
        temp = np.apply_along_axis(self.extract, 1, tweets.values.reshape(tweets.size, -1)).astype(np.int)
        return temp

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def extract(self, arr):
        return np.array([arr[0].count(mark) for mark in self.marks])
