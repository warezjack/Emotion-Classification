import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#add more emotions here, create directories by emotion label
emotions = [
	'joy',
	'anger',
	'sadness',
	'surprise',
	'fear',
	'disgust'
]

container_path = '/home/warez/watts/train';

load_dataset = sklearn.datasets.load_files(container_path, categories=emotions, shuffle=True, random_state=42)

#print load_dataset.target_names;
#print load_dataset.data;
#print load_dataset.filenames;

print load_dataset.target;

'''Document Preprocessing'''

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(load_dataset.data)
#print X_train_counts;
#print X_train_counts.shape;
#print count_vect.vocabulary_.get(u'algorithm');

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print X_train_tfidf.shape;

'''naive bayes classifier'''
clf = MultinomialNB().fit(X_train_tfidf, load_dataset.target)

#prediction
docs_new = ["Garner was perfect for this role, and shows his narrating skills as he explains how the two of them live. "]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, load_dataset.target_names[category]))



