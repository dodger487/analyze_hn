# Chris Riederer
# 2017-02-22

# Vectorify tags using Gensim, turn into lower dimensional space w/ LSI


from __future__ import print_function

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
import pandas as pd


FREQ_LIMIT = 1000  # Only keep tags that are used more than this number of times


logging.info("Begin read data")
tags = pd.read_csv("../data/ig_tag_data.csv")
logging.info("End read data")
print("Number of tags:", len(tags))
print(tags.head())

logging.info("concatenating tags")
tag_concat = tags.groupby("ig_user_id").tag.apply(list)
print("Number of users:", len(tag_concat))
print(tag_concat.head())

# Get the frequency of each tag
freq = tags.tag.value_counts()
logging.info("Highest frequency tags:")
print(freq.head())
print("Number of distinct tags:", len(freq))
high_freq = set(freq[freq > FREQ_LIMIT].index)

# Filter for only high frequency tags
tag_docs = [[tag for tag in user_tags if tag in high_freq]
                 for user_tags in tag_concat]

# Create dictionary, mapping words to ints
# Save it to disc
dictionary = corpora.Dictionary(tag_docs)
print("Dictionary size:", len(dictionary))
#logging.info("Saving dictionary")
dictionary.save("tag_dictionary_lda.pkl")
#logging.info("Done saving dictionary.")

# Create corpus, mapping a user's tags to a integer vectors
# Save it to disc
corpus = [dictionary.doc2bow(tag_list) for tag_list in tag_docs]
#tfidf = models.TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]
logging.info("Saving corpus.")
corpora.MmCorpus.serialize('tag_corpus.mm', corpus)
logging.info("Done saving corpus.")

#logging.info("Training LSI.")
#model = models.lsimodel.LsiModel(corpus_tfidf, id2word=dictionary)
#model.save("tags_lsi.gensim")

#logging.info("Trainding hi-dim LDA")
#model_hi = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=1000)
#model_hi.save("tags_lda_1000.gensim")

#model_lo = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=200)

