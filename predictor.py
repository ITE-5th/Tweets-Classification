import numpy as np
from sklearn.externals import joblib
import pandas as pd
from preprocessing.preprocessing import pre_process_tweet


class Predictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, tweets):
        tweets = np.array(tweets).astype(np.str)
        tweets = pd.DataFrame({"tweet": tweets})
        tweets = pre_process_tweet(tweets)
        predicted = self.model.predict(tweets)
        return ["yes" if pred >= 0.5 else "no" for pred in predicted]


if __name__ == '__main__':
    predictor = Predictor("models/predictor.pkl")
    tweets = ["أنا ضد قيادة المرأة"]
    p = predictor.predict(tweets)
    print(p)
