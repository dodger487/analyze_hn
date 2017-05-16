Code to classify Hacker News articles into a few categories.

Code is a bit messy at the moment, but there are roughly two phases:

- Phase 1: Train LDA to reduce article text to a 100 dimensional vector, using Gensim
- Phase 2: Train Logistic Regression or Random Forest on a labeled dataset to map vectors to topics.

