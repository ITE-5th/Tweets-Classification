from sklearn.base import TransformerMixin, BaseEstimator


class LengthTransformer(BaseEstimator, TransformerMixin):
    def transform(self, tweets):
        temp = tweets.apply(lambda x: len(x))
        return temp.values.reshape(temp.size, -1)

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
