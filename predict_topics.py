# Chris Riederer
# 2017-05-13

"""Predict topics using topic models or other stuff!"""

from os import listdir
from os.path import isfile, join
import pickle

from dplython import *
from gensim import corpora, models, utils
import pandas as pd
from sklearn import linear_model, ensemble, feature_extraction, cross_validation, metrics


label_df = DplyFrame(pd.read_csv("supervised_topics.csv"))

# TODO: load models etc.
# Older LDA, worked fine!
# dictionary = corpora.Dictionary.load("hn_dictionaryMay13_2152.pkl")
# lda = models.LdaModel.load("model_100topics_10passMay13_2159.gensim")

# dictionary = corpora.Dictionary.load("hn_dictionaryMay14_0005.pkl")
# lda = models.LdaModel.load("model_100topics_10passMay14_0018.gensim")

dictionary = corpora.Dictionary.load("hn_dictionaryMay14_0240.pkl")
lda = models.LdaModel.load("model_100topics_10passMay14_0259.gensim")


def label_article(text, trained_model):
  text = text.lower()
  tokens = list(utils.tokenize(text))
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


def makeTimeFilename(prefix, ext):
  """Creates a filename with the time in it."""  
  suffix = time.strftime("%b%d_%H%M") + ext
  return prefix + suffix


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


labels = np.array(labels)
dict_vectorizer = feature_extraction.DictVectorizer()

story_ids2 = np.array(list(story_id_to_topics.keys()))
data = np.array([[tdict.get(i, 0) for i in range(lda.num_topics)]
                  for tdict in story_id_to_topics.values()])

results = []
lr_models = {}
for label in set(labels):
  positive_story_ids = set(df >> sift(X["labels"] == label) >> X.story_id.values)
  y_ = np.array([s in positive_story_ids for s in story_ids2])
  X_ = data
  lr = linear_model.LogisticRegression(C=C)
  print(label, Counter(y_))

  cv_score = cross_validation.cross_val_score(
    lr, X_, y_, cv=10, scoring="roc_auc").mean()
  
  lr = lr.fit(X_, y_)
  lr_models[label] = lr
  probs = lr.predict_proba(X_)[:, 1]
  results.append({"alg": "log reg", "label": label, "auc": cv_score})
  print(C, label, cv_score, len(probs[probs > 0.19]), Counter(labels == label))
  print()
results_df = pd.DataFrame(results)

lr_fname = makeTimeFilename("models/logistic_models_", ".pkl")
print("writing file", lr_fname)
with open(lr_fname, "wb") as f:
  pickle.dump(lr_models, f, protocol=2)


results = []
rf_models = {}
for label in set(labels):
  positive_story_ids = set(df >> sift(X["labels"] == label) >> X.story_id.values)
  y_ = np.array([s in positive_story_ids for s in story_ids2])
  X_ = data
  rf = ensemble.RandomForestClassifier(n_estimators=200)

  cv_score = cross_validation.cross_val_score(
    rf, X_, y_, cv=10, scoring="roc_auc").mean()

  rf.fit(X_, y_)
  rf_models[label] = rf
  this_output = {"alg": "random forest", "label": label, "auc": cv_score}
  print(this_output)
  results.append(this_output)
  print()
results_df2 = pd.DataFrame(results)


rf_fname = makeTimeFilename("models/randomforest_models", ".pkl")
print("writing random forest file", rf_fname)
with open(rf_fname, "wb") as f:
  pickle.dump(rf_models, f, protocol=2)

