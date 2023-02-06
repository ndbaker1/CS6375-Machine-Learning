from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(data):
    return CountVectorizer().fit_transform(data).toarray()


def bernoulli(data):
    return CountVectorizer().fit_transform(data).toarray()
