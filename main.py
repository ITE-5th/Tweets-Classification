import random

import nltk


def tweet_features(x):
    try:
        temp = {"has_deny": any(c in x for c in ("نلغي", "يرفض", "إلغاء")),
                "has_no": any(c in x for c in "لا"),
                }
        return temp
    except:
        print(x)


labeled_tweets = []
# f = open("Driving_Data_Cleaned_with_hashtag.txt", "r", encoding="utf8")
# f = open("Driving_Data_Cleaned_without_hash.txt", "r", encoding="utf8")
f = open("tweetsData.txt", "r", encoding="utf8")

line_no = 0
for line in f:
    line_no = line_no + 1
    res = line.split('||')
    try:
        labeled_tweets.append((res[0], (res[1]).lower().strip(' \t\n\r')))
    except:
        print("Error in Data \n", res, "\n in Line", line_no)

random.shuffle(labeled_tweets)

data = [(tweet_features(x[0]), x[1]) for x in labeled_tweets]

train_data, test_data = data[:int(len(data) * 0.9)], data[int(len(data) * 0.9) + 1:]

su = 0
rr = 5
for i in range(rr):
    clz = nltk.NaiveBayesClassifier.train(train_data)
    su += nltk.classify.accuracy(clz, test_data)
su /= rr
print(su)
