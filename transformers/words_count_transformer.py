import string

import nltk
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class WordsCountTransformer(BaseEstimator, TransformerMixin):
    def transform(self, tweets):
        temp = np.apply_along_axis(self.extract, 1, tweets.values.reshape(tweets.size, -1)).reshape(tweets.size,
                                                                                                    -1).astype(np.int)
        return temp

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def extract(self, arr):
        return np.array([sum(1 for word in nltk.word_tokenize(arr[0]) if
                             word not in string.punctuation and word not in ["،", "..", "...", "؛"])])
