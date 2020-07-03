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

		cursor = documents.find({},{"authors": 1})

		vertices = []
		edges = []
		weights = []
		for c in cursor:
			for author in c["authors"]:
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
		self.partition = louvain.find_partition(self.g, louvain.ModularityVertexPartition);


	'''
	retrieve, for each parition:
	the number of nodes and the internal or external edges
	'''
	def getInternalExternalEdgesAndNodesNr(self):

		adjaceyMatrix = self.g.get_adjacency()

		internalEdgesAndNodesNr = {}

		for pindex in range(0, len(self.partition)):
			nodesNr = len(self.partition[pindex])

			if nodesNr > 1:
				nodeIds = [nodeId for nodeId in self.partition[pindex]]

				internalEdges = []
				externalEdges = []

				for nodeId in nodeIds:
					nodeCorrespondentIds = [i for i, x in enumerate(adjaceyMatrix[nodeId]) if x == 1 and i > nodeId]
					for nodeCorrespondentId in nodeCorrespondentIds:
						if nodeCorrespondentId in nodeIds:
							internalEdges.append((nodeId, nodeCorrespondentId))
						else:
							externalEdges.append((nodeId, nodeCorrespondentId))

				internalEdgesNr = len(internalEdges)
				externalEdgesNr = len(externalEdges)

				internalEdgesAndNodesNr[pindex] = (nodesNr, internalEdgesNr, externalEdgesNr)

		return internalEdgesAndNodesNr


	def computeInternalDensity(self):

		internalEdgesAndNodesNr = self.getInternalExternalEdgesAndNodesNr()
		internalDensities = {}

		for pindex in internalEdgesAndNodesNr:
			nodesNr = internalEdgesAndNodesNr[pindex][0]
			internalEdgesNr = internalEdgesAndNodesNr[pindex][1]

			internalDensities[pindex] = (internalEdgesNr) / (nodesNr * (nodesNr - 1)/2)

		avgInternalDensity = sum(internalDensities.values())/len(internalDensities)

		self.printGraph(internalDensities, {'outputFileName': 'internalDensity_' + str(self.dataset), \
			'title': 'Internal density for ' + str(self.dataset), \
			'ylabel': 'Internal density'})

		print("The average internal density is {}".format(avgInternalDensity))

	def computeAvgerageDegree(self):

		internalEdgesAndNodesNr = self.getInternalExternalEdgesAndNodesNr()
		avgerageDegrees = {}

		for pindex in internalEdgesAndNodesNr:
			nodesNr = internalEdgesAndNodesNr[pindex][0]
			internalEdgesNr = internalEdgesAndNodesNr[pindex][1]

			avgerageDegrees[pindex] = 2*(internalEdgesNr) / nodesNr

		avgAvgerageDegrees = sum(avgerageDegrees.values())/len(avgerageDegrees)

		self.printGraph(avgerageDegrees, {'outputFileName': 'averageDegree_' + str(self.dataset), \
			'title': 'Average degree for ' + str(self.dataset), \
			'ylabel': 'Average degree'})

		print("The average internal degree is {}".format(avgAvgerageDegrees))

	def computeExpansion(self):

		internalEdgesAndNodesNr = self.getInternalExternalEdgesAndNodesNr()
		expansions = {}

		for pindex in internalEdgesAndNodesNr:
			nodesNr = internalEdgesAndNodesNr[pindex][0]
			externalEdgesNr = internalEdgesAndNodesNr[pindex][2]

			expansions[pindex] = externalEdgesNr / nodesNr

		avgExpansion = sum(expansions.values())/len(expansions)

		print("The average expansion is {}".format(avgExpansion))

	def computeConductance(self):

		conductances = {}

		internalEdgesAndExternalEdgesNr = self.getInternalExternalEdgesAndNodesNr()
		for pindex in internalEdgesAndExternalEdgesNr:
			internalEdgesNr = internalEdgesAndExternalEdgesNr[pindex][1]
			externalEdgesNr = internalEdgesAndExternalEdgesNr[pindex][2]

			conductances[pindex] = externalEdgesNr / (2 * internalEdgesNr + externalEdgesNr)

		avgConductance = sum(conductances.values())/len(conductances)

		print("The average conductance is {}".format(avgConductance))

	def printGraph(self, datasetEvaluation, datasetSpecifics):

		plt.title(datasetSpecifics['title'])
		plt.xlabel('Community id')
		plt.ylabel(datasetSpecifics['ylabel'])

		plt.plot(list(datasetEvaluation.keys()), list(datasetEvaluation.values()), color='g')
		
		plt.savefig('figures/'+datasetSpecifics['outputFileName'])

		plt.clf()
		plt.cla()
		plt.close()


	def computeModularity(self):

		modularity = self.g.modularity(self.partition, weights=self.g.es['weight'])
		print("The modularity Q based on igraph is {}".format(modularity))
			

print("100 per batch")

arxivAuthorsEvaluationC_100 = ArxivAuthorsEvaluation('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationC_100.buildGraph()
arxivAuthorsEvaluationC_100.computeModularity()
# arxivAuthorsEvaluationC_100.computeInternalDensity()
arxivAuthorsEvaluationC_100.computeAvgerageDegree()
# arxivAuthorsEvaluationC_100.computeExpansion()
# arxivAuthorsEvaluationC_100.computeConductance()

arxivAuthorsEvaluationR_100 = ArxivAuthorsEvaluation('TR_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationR_100.buildGraph()
arxivAuthorsEvaluationR_100.computeModularity()
# arxivAuthorsEvaluationR_100.computeInternalDensity()
arxivAuthorsEvaluationR_100.computeAvgerageDegree()
# arxivAuthorsEvaluationR_100.computeExpansion()
# arxivAuthorsEvaluationR_100.computeConductance()

print("1000 per batch")

arxivAuthorsEvaluationC_1000 = ArxivAuthorsEvaluation('TE_1000', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationC_1000.buildGraph()
# arxivAuthorsEvaluationC_1000.computeModularity()
# arxivAuthorsEvaluationC_1000.computeInternalDensity()
arxivAuthorsEvaluationC_1000.computeAvgerageDegree()
# arxivAuthorsEvaluationC_1000.computeExpansion()
# arxivAuthorsEvaluationC_1000.computeConductance()


arxivAuthorsEvaluationR_1000 = ArxivAuthorsEvaluation('TR_1000', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationR_1000.buildGraph()
# arxivAuthorsEvaluationR_1000.computeModularity()
# arxivAuthorsEvaluationR_1000.computeInternalDensity()
arxivAuthorsEvaluationR_1000.computeAvgerageDegree()
# arxivAuthorsEvaluationR_1000.computeExpansion()
# arxivAuthorsEvaluationR_1000.computeConductance()