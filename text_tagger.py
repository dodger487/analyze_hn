from __future__ import print_function

import pickle

from gensim import corpora, models
import nltk
import numpy as np


class TextTagger(object):
  """Object which tags articles. Needs topic modeler and """
  def __init__(self, topic_modeler, gensim_dict, lr_dict, threshold=0.5):
    super(TextTagger, self).__init__()
    self.topic_modeler = topic_modeler
    self.gensim_dict = gensim_dict
    self.lr_dict = lr_dict
    self.threshold = threshold

  def text_to_topic_list(self, text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    bow = self.gensim_dict.doc2bow(tokens)
    return self.topic_modeler[bow]    

  def text_to_numpy(self, text):
    out = np.zeros(self.topic_modeler.num_topics)
    for idx, val in self.text_to_topic_list(text):
      out[idx] = val
    return out
    
  def text_to_topic_dict(self, text):
    return {topic: weight for topic, weight in self.label_article(text)}

  def text_to_tags(self, text):
    input_vect = np.array([self.text_to_numpy(text)])
    tags = []
    for label, lr_model in self.lr_dict.items():
      tag_prob = lr_model.predict_proba(input_vect)[0, 1]
      if tag_prob > self.threshold:
        tags.append(label)
    return tags

  @classmethod
  def init_from_files(cls, topic_model_fname, gensim_dict_fname, lr_dict_fname,
                      *args, **kwargs):
    topic_modeler = models.ldamodel.LdaModel.load(topic_model_fname)
    gensim_dict = corpora.Dictionary.load(gensim_dict_fname)
    with open(lr_dict_fname, "rb") as f:
      lr_dict = pickle.load(f)
    return cls(topic_modeler, gensim_dict, lr_dict, *args, **kwargs)