from igraph import *
import pymongo
import pprint
from itertools import combinations
import louvain
import collections
import matplotlib.pyplot as plt
from evaluation import ArxivAuthorsEvaluation

class ArxivAuthorsLouvain():

	def __init__(self, dataset, dbName):
		
		self.dataset = dataset
		self.dbName = dbName
		self.g = Graph()
		self.partition = None

	def buildGraph(self):

		print("Dataset" + str(self.dataset) + " ====================================")

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		cursor = documents.find({},{"authors": 1})

		vertices = []
		edges = []
		weights = []
		for c in cursor:
			for author in c["authors"]:
				if (author not in vertices):
					vertices.append(author)

			for pair in combinations(c["authors"], 2):
				ind = edges.index(pair) if pair in edges else -1
				if ind == -1:
					edges.append(pair)
					weights.append(1)
				else:
					weights[ind] += 1

		self.g.add_vertices(vertices)
		self.g.add_edges(edges)
		self.g.es['weight'] = weights
		self.partition = louvain.find_partition(self.g, louvain.ModularityVertexPartition)

	def evaluateGraph(self):

		evaluationLib = ArxivAuthorsEvaluation(self.g, self.partition, self.dataset)
		evaluationLib.computeModularity()
		evaluationLib.computeInternalDensity()
		evaluationLib.computeAvgerageDegree()
		evaluationLib.computeExpansion()
		evaluationLib.computeConductance()


print("100 per batch")
arxivAuthorsEvaluationTE_100 = ArxivAuthorsLouvain('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationTE_100.buildGraph()
arxivAuthorsEvaluationTE_100.evaluateGraph()

