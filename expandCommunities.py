from igraph import *
import pymongo
import pprint
from itertools import combinations
import louvain
import collections
import matplotlib.pyplot as plt
import multiprocessing as mp
from evaluation import ArxivAuthorsEvaluation

class ArxivAuthorsCommunityExpansion():

	def __init__(self, dataset, dbName):
		
		self.dataset = dataset
		self.dbName = dbName
		self.g = None
		self.partition = collections.defaultdict(list)

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

		self.partition = VertexClustering(self.g, self.g.vs['partitionId'])

	def evaluateGraph(self):

		evaluationLib = ArxivAuthorsEvaluation(self.g, self.partition, self.dataset)
		evaluationLib.computeModularity()
		evaluationLib.computeInternalDensity()
		evaluationLib.computeAvgerageDegree()
		evaluationLib.computeExpansion()
		evaluationLib.computeConductance()
				

print('100 per batch')

arxivAuthorsEvaluationTE_100 = ArxivAuthorsCommunityExpansion('TE_100', 'ArxivCommunityDetectionDatasets')
arxivAuthorsEvaluationTE_100.buildGraph()
arxivAuthorsEvaluationTE_100.evaluateGraph()