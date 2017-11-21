import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureAmp(BaseEstimator, TransformerMixin):

    def transform(self, tweets):
        return tweets

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
