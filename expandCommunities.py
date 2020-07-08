from igraph import *
import pymongo
import pprint
from itertools import combinations
import louvain
import collections
import matplotlib.pyplot as plt
import multiprocessing as mp
from evaluation import ArxivAuthorsEvaluation
import copy
from random import randint

class ArxivAuthorsCommunityExpansion():

	def __init__(self, dataset, dbName):
		
		self.dataset = dataset
		self.dbName = dbName
		self.g = None
		self.partition = collections.defaultdict(list)


	def computeClusteringEdges(self, mpInput):

		authorsInSameCluster = mpInput[0]
		helperG = mpInput[1]
		clusterId = mpInput[2]
		
		clusteringEdges = []
		clusteringWeights = []

		edges = [edge.tuple for edge in helperG.es]

		partitions2nodes = collections.defaultdict(list)
		nodeIds2clusters = {}

		for node in helperG.vs:
			partitions2nodes[node['partitionId']].append(node.index)
			nodeIds2clusters[node.index] = node['clusterId']

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
						clusteringEdges.append((helperG.vs[pair[0]]['name'], helperG.vs[pair[1]]['name']))

		helperG.add_edges(clusteringEdges)
		helperG.es[len(edges):]['weight'] = [1] * len(clusteringEdges)

		edges = [edge.tuple for edge in helperG.es]

		# sort the partition dictionary ascending by size
		partitions2nodes = {k: v for k, v in sorted(partitions2nodes.items(), key = lambda item: len(item[1]))}
		partitions2nodes = {k: v for k, v in sorted(partitions2nodes.items(), key = lambda item: item[0])}


		# filter partitions
		# only partitions which contain all nodes from this cluster

		legitPartitions = {}

		for partitionId in partitions2nodes:
			nodesSameClusterInPartition = list(set([author['nodeId'] for author in authorsInSameCluster]) & set(partitions2nodes[partitionId]))
			if (len(nodesSameClusterInPartition) != len(partitions2nodes[partitionId])):
				continue
			legitPartitions[partitionId] = partitions2nodes[partitionId]

		
		# inter partitions
		'''
		We augument the inter partition edges by uniting the nodes FROM DIFFERENT PARTITIONS
		which belong to the same cluster (determined by clustering the articles using the Spherical K-Means algorithm)
		We only keep the partition unions which maximize the modularity
		'''

		updatedPartitions = {}
		isChange = True;

		while (isChange == True):

			isChange = False

			maxPartition1 = None
			maxPartition2 = None

			maxClusteringEdges = []

			maxModularity = helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight'])

			for partitionId1 in list(legitPartitions):

				if (partitionId1 not in legitPartitions):
					continue

				for partitionId2 in list(legitPartitions):
					
					if (partitionId1 < partitionId2 and partitionId2 in legitPartitions):

						# merge partitions
						for nodeId in legitPartitions[partitionId2]:
							helperG.vs[nodeId]['partitionId'] = partitionId1

						nodeId1 = legitPartitions[partitionId1][randint(0, len(legitPartitions[partitionId1]) - 1)]
						nodeId2 = legitPartitions[partitionId2][randint(0, len(legitPartitions[partitionId2]) - 1)]

						helperG.add_edge(helperG.vs[nodeId1]['name'], helperG.vs[nodeId2]['name'], weight = 1)

						newModularity = helperG.modularity(helperG.vs['partitionId'], weights = helperG.es['weight'])

						if (newModularity > maxModularity):
							maxModularity = newModularity
							maxClusteringEdges = [(helperG.vs[nodeId1]['name'], helperG.vs[nodeId2]['name'])]
							maxPartition1 = partitionId1
							maxPartition2 = partitionId2

						helperG.delete_edges((helperG.vs[nodeId1]['name'], helperG.vs[nodeId2]['name']))
						# UNmerge partitions
						for nodeId in legitPartitions[partitionId2]:
							helperG.vs[nodeId]['partitionId'] = partitionId2

			if (maxPartition1 != None and maxPartition2 != None):
				print('MAX MOD = ', maxModularity)
				isChange = True
				clusteringEdges += maxClusteringEdges
				for nodeId in legitPartitions[maxPartition2]:
					updatedPartitions[nodeId] = maxPartition1
				legitPartitions[maxPartition1] += legitPartitions[maxPartition2]
				del legitPartitions[maxPartition2]


		# print('CLUSTER ID ', clusterId, updatedPartitions)

		clusteringWeights = [1] * (len(clusteringEdges) - 1) 

		return (clusteringEdges, clusteringWeights, updatedPartitions)

	def buildGraph(self):

		print('Dataset' + str(self.dataset) + ' ====================================')

		self.g = Graph()
		helperG = Graph()

		client = pymongo.MongoClient(host='localhost', port=27017)
		db = client[self.dbName]
		documents = db[self.dataset]

		cursor = documents.find({},{'authors': 1, 'clusterId': 1}).limit(100)

		vertices = []
		edges = []
		weights = []
		clusterIds = []
		clusterIds2nodeIds = collections.defaultdict(list)

		nodeIdx = 0

		for c in cursor:
			for author in c['authors']:
				if (author not in vertices):
					vertices.append(author)
				clusterIds.append(c['clusterId'])
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

		for node in helperG.vs:
			clusterIds2nodeIds[node['clusterId']].append({'author': node['name'], 'nodeId': node.index})

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

		mpInput = [(clusterIds2nodeIds[clusterId], helperG, clusterId) for clusterId in list(clusterIds2nodeIds.keys())]

		results = pool.map(self.computeClusteringEdges, mpInput)

		updatedPartitions = {}

		for result in results:
			edges += result[0]
			weights += result[1]
			updatedPartitions.update(result[2])

		print('Edges after augumentation ' + str(len(edges)))
			
		self.g.add_vertices(vertices)
		self.g.add_edges(edges)
		self.g.es['weight'] = weights

		for finalNode in self.g.vs:
			self.g.vs[finalNode.index]['partitionId'] = updatedPartitions[finalNode.index] \
															if finalNode.index in updatedPartitions \
															else helperG.vs[finalNode.index]['partitionId']

		self.partition = VertexClustering(self.g, self.g.vs['partitionId'])

		plot(self.g)

		plot(self.partition)

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