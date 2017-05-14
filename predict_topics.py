# Chris Riederer
# 2017-05-13

"""Predict topics using topic models or other stuff!"""

from os import listdir
from os.path import isfile, join

from dplython import *
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn import linear_model, feature_extraction, cross_validation, metrics


label_df = DplyFrame(pd.read_csv("supervised_topics.csv"))

ARTICLE_PATH = "../scrape_hn/stories/"
fnames = [join(ARTICLE_PATH, f) for f in listdir(ARTICLE_PATH) 
              if isfile(join(ARTICLE_PATH, f))]


# TODO: load models etc.

def label_article(text, trained_model):
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  bow = dictionary.doc2bow(tokens)
  return trained_model[bow]


def article_to_dict(text, trained_model):
  return {topic: weight for topic, weight in label_article(text, trained_model)}


def story_id_to_topicdict(story_id, trained_model=model_hi):
  """Given an ID for a story, read in the data and return the LDA topics as a
  dictionary of topic_id -> weight.
  """
  with open(join(ARTICLE_PATH, str(story_id) + ".txt")) as f:
    text = f.read()

  return article_to_dict(text, trained_model)

labels = []
feature_dict = []
story_ids = []
for label, story_id in zip(label_df.topic, label_df.id):
  if isfile(join(ARTICLE_PATH, str(story_id) + ".txt")):
    labels.append(label)
    feature_dict.append(story_id_to_topicdict(story_id))
    story_ids.append(story_id)
    print(label)

labels = np.array(labels)
X_ = feature_extraction.DictVectorizer().fit_transform(feature_dict)


results = []
for label in set(labels):
  for C in [1]:
  # for C in [0.01, 0.1, 1, 10]:
    lr = linear_model.LogisticRegression(C=C)
    cv_score = cross_validation.cross_val_score(
      lr, X_, labels == label, cv=10, scoring="roc_auc").mean()
    lr = lr.fit(X_, labels == label)
    probs = lr.predict_proba(X_)[:, 1]
    # print(label, )
    # metrics.confusion_matrix(labels == label, lr.fit())
    # results.append({"C": C, "label": label, "auc": cv_score})
    print(C, label, len(probs[probs > 0.2]), Counter(labels == label))
  print()

results_df = DplyFrame(results)
