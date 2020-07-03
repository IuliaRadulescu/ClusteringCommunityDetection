from igraph import *
import pymongo
import pprint
from itertools import combinations

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from spherecluster import spherical_kmeans
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from scipy import sparse


class ArxivAuthorsClustering:

	def __init__(self, dataset, dbName):
	
		self.dataset = dataset
		self.dbName = dbName
		self.partition = None
		self.arxivArticles = None

	def readInformationFromDataset(self):

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		cursor = documents.find({},{'_id': 1, 'authors': 1, 'summary': 1, 'title': 1, 'real_category': 1})

		self.arxivArticles = []

		for c in cursor:
			self.arxivArticles.append({'_id': c['_id'], 'title': c['title'], 'authors': c['authors'], 'summary': c['summary'], 'category': c['real_category']})

		client.close()

	def preprocessSummaries(self):

		nltk.download('stopwords')
		nltk.download('punkt')
		nltk.download('wordnet')

		lemmatizer = WordNetLemmatizer()

		for record in self.arxivArticles:
			stop = stopwords.words('english') + list(string.punctuation)
			tokens = [i for i in word_tokenize(record['summary'].lower()) if i not in stop]

			record['summary'] = [lemmatizer.lemmatize(token) for token in tokens]

	def clusterSummaries(self, doc2vec = True):

		docs = []
		docLabel = {}
		categories = {}

		k = 0
		for record in self.arxivArticles:
		    if (record['category'] in categories):
		        docLabel[str(record['_id'])] = categories[record['category']]
		    else:
		        categories[record['category']] = k
		        docLabel[str(record['_id'])] = k
		        k = k+1

		labels = np.array(list(docLabel.values()))

		nrClusters = len(set(labels))

		if (doc2vec == True):

			summaries = [record['summary'] for record in self.arxivArticles]

			taggedDocuments = [TaggedDocument(summary, [i]) for i, summary in enumerate(summaries)]
			model = Doc2Vec(vector_size = 16, window = 3, min_count = 2, cbow = 1, workers = 4, epochs = 900)

			model.build_vocab(taggedDocuments)

			model.train(taggedDocuments, total_examples=len(taggedDocuments), epochs=900)

			X = []

			for summary in summaries:
				X.append(model.infer_vector(summary))

			X = np.array(X)

		else:

			summaries = [' '.join(record['summary']) for record in self.arxivArticles]

			#instantiate CountVectorizer()
			cv = CountVectorizer()
			 
			# this steps generates word counts for the words in your docs
			word_count_vector = cv.fit_transform(summaries)

			print(np.shape(word_count_vector))

			tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
			tfidf_transformer.fit(word_count_vector)
			 
			# tf-idf scores
			X = tfidf_transformer.transform(word_count_vector)
		
		skm = spherical_kmeans.SphericalKMeans(n_clusters=5, init='k-means++', n_init=20)
		skmLabels = skm.fit(X).labels_.tolist()

		# self.writeClusters2DB(skmLabels)

		# self.printClusters(skmLabels)

		print(metrics.homogeneity_score(labels, skmLabels))
		print(metrics.completeness_score(labels, skmLabels))
		print(metrics.v_measure_score(labels, skmLabels))
		print(metrics.adjusted_rand_score(labels, skmLabels))
		print(metrics.adjusted_mutual_info_score(labels, skmLabels))


	def printClusters(self, labels):

		print("Dataset" + str(self.dataset) + " ====================================")

		g = Graph()

		vertices = []
		edges = []
		weights = []
		for c in self.arxivArticles:
			for author in c["authors"]:
				vertices.append(author)

			for pair in combinations(c["authors"], 2):
				ind = edges.index(pair) if pair in edges else -1
				if ind == -1:
					edges.append(pair)
					weights.append(1)
				else:
					weights[ind] += 1

		palette = ClusterColoringPalette(len(set(labels)))
		colors = [palette[label] for label in labels]

		g.add_vertices(vertices)
		g.add_edges(edges)
		g.es['weight'] = weights
		g.vs["color"] = colors

		plot(g)

	def writeClusters2DB(self, kmeansLabels):

		def addClusterId(x, y):
			x['clusterId'] = y
			return x

		self.arxivArticles = list(map(addClusterId, self.arxivArticles, kmeansLabels))

		print(self.arxivArticles[0])

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		for record in self.arxivArticles:
			print(record)
			documents.update({'_id': record['_id']}, {'$set': {'clusterId': record['clusterId']}}, False, False)

		client.close()


arxivAuthorsClustering = ArxivAuthorsClustering('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsClustering.readInformationFromDataset()
arxivAuthorsClustering.preprocessSummaries()
arxivAuthorsClustering.clusterSummaries(True)
