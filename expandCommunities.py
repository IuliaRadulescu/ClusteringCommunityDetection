from igraph import *
import pymongo
import pprint
from itertools import combinations
import louvain
import collections
import matplotlib.pyplot as plt
import multiprocessing as mp

class ArxivAuthorsEvaluation():

	def __init__(self, dataset, dbName):
		
		self.dataset = dataset
		self.dbName = dbName
		self.g = None
		self.partition = None

	def computeClusteringEdges(self, mpInput):

		authorsInSameCluster = mpInput[0]
		helperG = mpInput[1]
		
		clusteringEdges = []
		clusteringWeights = []

		edges = [edge.tuple for edge in helperG.es]

		partitions2nodes = collections.defaultdict(list)

		for node in helperG.vs:
			partitions2nodes[node['partitionId']].append(node.index)

		oldModularity = helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight'])

		# intra partitions augumented edges
		'''
		We augument the intra partition edges by uniting the nodes
		which belong to the same cluster (determined by clustering the articles using the Spherical K-Means algorithm)
		'''
		for partitionId in partitions2nodes:
			nodesSameClusterPerPartition = list(set([author['nodeId'] for author in authorsInSameCluster]) & set(partitions2nodes[partitionId]))

			if (len(nodesSameClusterPerPartition) > 1):
				for pair in combinations(nodesSameClusterPerPartition, 2):
					if ((pair[0], pair[1]) not in edges and (pair[1], pair[0]) not in edges):
						helperG.add_edge(helperG.vs[pair[0]], helperG.vs[pair[1]], weight = 1)
						clusteringEdges.append((helperG.vs[pair[0]]['name'], helperG.vs[pair[1]]['name']))
			
			newModularity = helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight'])

		edges = [edge.tuple for edge in helperG.es]

		# inter partitions
		'''
		We augument the inter partition edges by uniting the nodes FROM DIFFERENT PARTITIONS
		which belong to the same cluster (determined by clustering the articles using the Spherical K-Means algorithm)
		We only keep the partition unions which maximize the modularity
		'''
		for partitionId1 in partitions2nodes:
			
			nodesSameClusterInPartition1 = list(set([author['nodeId'] for author in authorsInSameCluster]) & set(partitions2nodes[partitionId1]))
			
			for partitionId2 in partitions2nodes:
				
				if (partitionId1 < partitionId2):

					nodesSameClusterInPartition2 = list(set([author['nodeId'] for author in authorsInSameCluster]) & set(partitions2nodes[partitionId2]))
					nodesToCombine = nodesSameClusterInPartition1 + nodesSameClusterInPartition2
					
					candidateEdges = [] # the edges resulting by uniting nodes in the same cluster, from different partitions
					# not all candidate edges make it into the final configuration

					for pair in combinations(nodesToCombine, 2):
						if ((pair[0], pair[1]) not in edges and (pair[1], pair[0])):
							helperG.add_edge(helperG.vs[pair[0]]['name'], helperG.vs[pair[1]]['name'], weight = 1)
							candidateEdges.append((helperG.vs[pair[0]]['name'], helperG.vs[pair[1]]['name']))

					newModularity = helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight'])

					# if new configuration maximizes modularity, keep it
					if (newModularity > oldModularity):
						print('newModularity = ' + str(newModularity))
						oldModularity = newModularity
						clusteringEdges += candidateEdges
					else:
						# if new configuration does not bring any benefit, drop it
						helperG.delete_edges(candidateEdges)

		clusteringWeights = [1] * (len(clusteringEdges) - 1) 

		return (clusteringEdges, clusteringWeights)

	def buildGraph(self):

		print('Dataset' + str(self.dataset) + ' ====================================')

		self.g = Graph()
		helperG = Graph()

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		cursor = documents.find({},{'authors': 1, 'clusterId': 1})

		vertices = []
		edges = []
		weights = []
		clusterIds = []
		clusterIds2nodeIds = collections.defaultdict(list)

		nodeIdx = 0

		for c in cursor:
			for author in c['authors']:
				vertices.append(author)
				clusterIds.append(c['clusterId'])
				clusterIds2nodeIds[c['clusterId']].append({'author': author, 'nodeId': nodeIdx})
				nodeIdx += 1

			for pair in combinations(c['authors'], 2):
				ind = edges.index(pair) if pair in edges else -1
				if ind == -1:
					edges.append(pair)
					weights.append(1)
				else:
					weights[ind] += 1

		helperG.add_vertices(vertices)
		helperG.add_edges(edges)
		helperG.es['weight'] = weights
		helperG.vs['clusterId'] = clusterIds

		helperPartition = louvain.find_partition(helperG, louvain.ModularityVertexPartition)

		nodes2communities = {}

		for partitionId in range(0, len(helperPartition)):
			nodeIds = helperPartition[partitionId]
			for nodeId in nodeIds:
				helperG.vs[nodeId]['partitionId'] = partitionId

		# clustering modularity
		print(helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight']))

		print('Edges before augumentation ' + str(len(edges)))

		# add augumented edges
		# for each cluster in paralel
		pool = mp.Pool(mp.cpu_count())

		mpInput = [(clusterIds2nodeIds[clusterId], helperG) for clusterId in list(clusterIds2nodeIds.keys())]

		results = pool.map(self.computeClusteringEdges, mpInput)

		for result in results:
			edges += result[0]
			weights += result[1]

		print('Edges after augumentation ' + str(len(edges)))
			
		self.g.add_vertices(vertices)
		self.g.add_edges(edges)
		self.g.es['weight'] = weights

		for nodeId in [node.index for node in helperG.vs]:
			self.g.vs[nodeId]['partitionId'] = helperG.vs[nodeId]['partitionId']

		self.partition = helperPartition

		# plot(self.g)

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

		modularity = self.g.modularity(self.g.vs['partitionId'], weights = self.g.es['weight'])
		print("The modularity Q based on igraph is {}".format(modularity))
				

print('100 per batch')

arxivAuthorsEvaluationC_100 = ArxivAuthorsEvaluation('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationC_100.buildGraph()
arxivAuthorsEvaluationC_100.computeModularity()
arxivAuthorsEvaluationC_100.computeInternalDensity()
arxivAuthorsEvaluationC_100.computeAvgerageDegree()
arxivAuthorsEvaluationC_100.computeExpansion()
arxivAuthorsEvaluationC_100.computeConductance()