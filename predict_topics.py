# Chris Riederer
# 2017-05-13

"""Predict topics using topic models or other stuff!"""

from os import listdir
from os.path import isfile, join
import pickle

from dplython import *
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn import linear_model, feature_extraction, cross_validation, metrics

label_df = DplyFrame(pd.read_csv("supervised_topics.csv"))

# ARTICLE_PATH = "../scrape_hn/stories/"
# fnames = [join(ARTICLE_PATH, f) for f in listdir(ARTICLE_PATH) 
#               if isfile(join(ARTICLE_PATH, f))]

# TODO: load models etc.
dictionary = corpora.Dictionary.load("hn_dictionaryMay13_2152.pkl")
lda = models.ldamodel.LdaModel.load("model_100topics_10passMay13_2159.gensim")


def label_article(text, trained_model):
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  bow = dictionary.doc2bow(tokens)
  return trained_model[bow]


def article_to_dict(text, trained_model):
  return {topic: weight for topic, weight in label_article(text, trained_model)}


def story_id_to_topicdict(story_id, trained_model=lda):
  """Given an ID for a story, read in the data and return the LDA topics as a
  dictionary of topic_id -> weight.
  """
  with open(join(ARTICLE_PATH, str(story_id) + ".txt")) as f:
    text = f.read()

  return article_to_dict(text, trained_model)

labels = []
topic_dicts = []
story_ids = []
story_id_to_topics = {}
for label, story_id in zip(label_df.topic, label_df.id):
  if isfile(join(ARTICLE_PATH, str(story_id) + ".txt")):
    labels.append(label)
    topic_dict = story_id_to_topicdict(story_id)
    topic_dicts.append(topic_dict)
    story_ids.append(story_id)
    story_id_to_topics[story_id] = topic_dict
    print(label)

df = DplyFrame(pd.DataFrame({"labels": labels, 
                             "story_id": story_ids, 
                             "features": topic_dicts}))

labels = np.array(labels)
dict_vectorizer = feature_extraction.DictVectorizer()

story_ids2 = np.array(list(story_id_to_topics.keys()))
data = dict_vectorizer.fit_transform(story_id_to_topics.values())

results = []
models = {}
for label in set(labels):
  positive_story_ids = set(df >> sift(X["labels"] == label) >> X.story_id.values)
  y_ = np.array([s in positive_story_ids for s in story_ids2])
  X_ = data
  lr = linear_model.LogisticRegression(C=C)
  cv_score = cross_validation.cross_val_score(
    lr, X_, y_, cv=10, scoring="roc_auc").mean()
  lr = lr.fit(X_, y_)
  models[label] = lr
  probs = lr.predict_proba(X_)[:, 1]
  print(C, label, cv_score, len(probs[probs > 0.2]), Counter(labels == label))
  print()

with open("models/logistic_models.pkl", "wb") as f:
  pickle.dump(models, f)

results_df = DplyFrame(results)
