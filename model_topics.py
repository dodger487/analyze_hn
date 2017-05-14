# GRRR Enterprises
# 2017-05-13

"""Topic modeling for Hacker News articles."""

from __future__ import print_function

from collections import Counter
import itertools
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
from os import listdir
from os.path import isfile, join
import string
import time

from gensim import corpora, models, utils
import nltk
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
ARTICLE_PATH = "../scrape_hn/stories/"


def makeTimeFilename(prefix, ext):
  """Creates a filename with the time in it."""  
  suffix = time.strftime("%b%d_%H%M") + ext
  return prefix + suffix


# Load data
fnames = [join(ARTICLE_PATH, f) for f in listdir(ARTICLE_PATH) 
              if isfile(join(ARTICLE_PATH, f))]
article_texts = []
logging.info("Reading articles...")
for fname in fnames:
  try:
    with open(fname) as f:
      text = f.read()
      article_texts.append(text)
  except Exception as e:
    print(fname, e)

# Convert to list of tokens
article_texts = [a.lower() for a in article_texts]
token_list = [list(utils.tokenize(article_text)) for article_text in article_texts]

# Get high frequency words
token_counts = Counter(itertools.chain(*token_list))

# Remove stopwords and punctuation
for word in STOPWORDS:
  del token_counts[word]
for word in string.punctuation:
  del token_counts[word]
del token_counts["|||"]

# Filter for only high frequency tags and stopwords
high_freq = set(token for token, count in token_counts.most_common()[:10000])
token_docs = [[token for token in article_tokens if token in high_freq]
                 for article_tokens in token_list]
# Remove empty documents
token_docs = [doc for doc in token_docs if len(doc) > 0]

# Make Gensim dictionary
dictionary = corpora.Dictionary(token_docs)
dict_fname = makeTimeFilename("hn_dictionary", ".pkl")
dictionary.save(dict_fname)

# Create corpus for topic model training
corpus = [dictionary.doc2bow(doc) for doc in token_docs]

# Train LDA
# model_hi = models.LdaMulticore(
#     corpus, id2word=dictionary, num_topics=100, passes=4, workers=2)
model_hi = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=10)
model_fname = makeTimeFilename("model_100topics_10pass", ".gensim")
model_hi.save(model_fname)


def label_article(text, trained_model):
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  bow = dictionary.doc2bow(tokens)
  return trained_model[bow]


def show_topics(text, trained_model, n=20):
  topics_and_weights = label_article(text, trained_model)
  for topic, weight in sorted(topics_and_weights, key=lambda x: -x[1]):
    print(weight, topic, trained_model.print_topic(topic, n))
    print()


