from igraph import *
import pymongo
import pprint
from itertools import combinations
import louvain
import collections
import matplotlib.pyplot as plt
'''
C - cs
R - random
E - egal
'''

class ArxivAuthorsEvaluation():

	def __init__(self, dataset, dbName):
		
		self.dataset = dataset
		self.dbName = dbName
		self.g = None
		self.partition = None

	def buildGraph(self):

		print("Dataset" + str(self.dataset) + " ====================================")

		self.g = Graph()

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		cursor = documents.find({},{"authors": 1, "clusterId": 1})

		vertices = []
		edges = []
		weights = []
		clusterIds2nodeIds = collections.defaultdict(list)

		for c in cursor:
			for author in c["authors"]:
				vertices.append(author)
				clusterIds2nodeIds[c["clusterId"]].append(author)

			for pair in combinations(c["authors"], 2):
				ind = edges.index(pair) if pair in edges else -1
				if ind == -1:
					edges.append(pair)
					weights.append(1)
				else:
					weights[ind] += 1

		# add augumented edges
		for clusterId in clusterIds2nodeIds:
			authorsInSameCluster = clusterIds2nodeIds[clusterId]
			for pair in combinations(authorsInSameCluster, 2):
				if (pair not in edges):
					edges.append(pair)
					weights.append(0)

		self.g.add_vertices(vertices)
		self.g.add_edges(edges)
		self.g.es['weight'] = weights


		self.partition = louvain.find_partition(self.g, louvain.ModularityVertexPartition)

		visual_style = {}

		# Set bbox and margin
		visual_style["bbox"] = (3000,3000)
		visual_style["margin"] = 17

		# Set vertex size
		visual_style["vertex_size"] = 20

		# Don't curve the edges
		visual_style["edge_curved"] = False

		# Set the layout
		my_layout = self.g.layout_drl()
		visual_style["layout"] = my_layout

		# plot(self.partition, 'Milly.eps', **visual_style)

	def computeModularity(self):

		modularity = self.g.modularity(self.partition, weights=self.g.es['weight'])
		print("The modularity Q based on igraph is {}".format(modularity))
				

		

print("100 per batch")

arxivAuthorsEvaluationC_100 = ArxivAuthorsEvaluation('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationC_100.buildGraph()
arxivAuthorsEvaluationC_100.computeModularity()
# arxivAuthorsEvaluationC_100.expandCommunities()