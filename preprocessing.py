import re

import numpy as np
from pyarabic.araby import strip_tashkeel


def pre_process(data):
    data = data.dropna(how="any")
    data.loc[:, "sentiment"] = data.loc[:, "sentiment"].apply(lambda x: int(x.lower().strip() == "yes")).astype(np.int)
    data = pre_process_tweet(data)
    data = data.drop_duplicates(subset="tweet")
    return data


def pre_process_tweet(data):
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: x.strip())
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: strip_tashkeel(x))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: re.sub("[أإآ]", "ا", x))
    # TODO: are we sure that we should delete all the tweets with any english letter?
    data = data.loc[~data.loc[:, "tweet"].str.contains("[a-zA-Z]"), :]
    return data