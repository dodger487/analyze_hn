def examine_articles(text_tagger, articles, num_articles=50, num_char=200):
  for idx, article_text in enumerate(articles[:num_articles]):
    if len(article_text) >= 20:
      print(idx, article_text[:num_char].replace("\n", ""))
      print(text_tagger.text_to_tags(article_text))
      print()



topic_info = [" ".join([str(i)] + [str(x[0]) for x in lda.show_topic(i)]) for i in range(100)]
def analyze_lr_model(lr, lda):
  coefs = lr.coef_[0]
  coefs_and_labels = sorted(list(zip(coefs, topic_info)), key=lambda x: -x[0]**2)
  return coefs_and_labels

def print_analyze(lr, lda, n=100):
  for i in analyze_lr_model(lr, lda)[:n]: print(i)

def print_lr_all(lr_models, n=100):
  for k, v in lr_models.items():
    print(k)
    print_analyze(v, "", n=n)
    print()


def label_article(text, trained_model):
  text = text.lower()
  tokens = list(utils.tokenize(text))
  bow = dictionary.doc2bow(tokens)
  return trained_model[bow]


def article_to_dict(text, trained_model):
  return {topic: weight for topic, weight in label_article(text, trained_model)}
