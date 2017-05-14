def examine_articles(text_tagger, articles, num_articles=50, num_char=200):
  for idx, article_text in enumerate(articles[:num_articles]):
    if len(article_text) >= 20:
      print(idx, article_text[:num_char].replace("\n", ""))
      print(text_tagger.text_to_tags(article_text))
      print()

topic_info = [" ".join([str(x[0]) for x in lda.show_topic(i)]) for i in range(100)]
def analyze_lr_model(lr, lda):
  coefs = lr.coef_[0]
  coefs_and_labels = sorted(list(zip(coefs, topic_info)), key=lambda x: -x[0]**2)
  return coefs_and_labels

def print_analyze(lr, lda, n=100):
  for i in analyze_lr_model(lr, lda)[:n]: print(i)
