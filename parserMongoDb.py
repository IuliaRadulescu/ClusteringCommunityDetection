import pymongo
import requests
import xml.etree.ElementTree as ET
import urllib.request as libreq
import random

client = pymongo.MongoClient(host='localhost', port=27017)

clusterCategories = [
	'cs.DB',
	'physics.geo-ph',
	'math.MG',
	'stats.ML',
	'q-bio.GN',
	'q-fin.EC'
]

batchSize = [10, 50]
datasetName = ['TR_', 'TE_']

# batchSize = [10]
# datasetName = ['TE_']

batchesNumber = 10

for d in range(0, len(batchSize)):
	print (batchSize[d])
	for j in range(0, len(datasetName)):

		# initialize batch start position for each dataset
		start = {}
		for clusterCategory in clusterCategories:
			start[clusterCategory] = 0

		collection = datasetName[j] + str(batchesNumber * batchSize[d])
		db = client.ArxivCommunityDetectionDatasets
		
		print(db)
		print(collection)
		
		documents = db[collection]
		documentList = []
		
		for k in range(0, len(clusterCategories)):
			
			for pos in range(0, batchesNumber * batchSize[d], batchSize[d]):

				print(pos)

				categoryId = k
				
				# if dataset TR, pick a cluster category at random and extract batchsize articles
				if (datasetName[j] == 'TR_'):
					categoryId = random.randint(0, len(clusterCategories)-1)

				category = clusterCategories[categoryId]
				
				with libreq.urlopen('http://export.arxiv.org/api/query?search_query=cat:'+ category +'&sortBy=submittedDate&sortOrder=descending&max_results=' + str(batchSize[d]) + '&start=' + str(start[clusterCategories[k]]) + '&max_results=' + str(batchSize[d])) as url:
					
					r = url.read()
				
					root = ET.fromstring(r)
					
					for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
						authors = []
						categories = []
						links = []
						title = entry.find('{http://www.w3.org/2005/Atom}title').text
						summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
						published = entry.find('{http://www.w3.org/2005/Atom}published').text
						for entry2 in entry.findall('{http://www.w3.org/2005/Atom}author'):
							authors.append(entry2.find('{http://www.w3.org/2005/Atom}name').text)
						journal_ref = entry.find('{http://arxiv.org/schemas/atom}journal_ref')
						if (journal_ref != None):
							journal_ref = journal_ref.text
						updated = entry.find('{http://www.w3.org/2005/Atom}updated').text
						for entry2 in entry.findall('{http://www.w3.org/2005/Atom}category'):
							categories.append(entry2.get('term'))
						for entry2 in entry.findall('{http://www.w3.org/2005/Atom}link'):
							links.append(entry2.get('href'))
					
						document = {
								"title": title,
								"summary": summary,
								"published": published,
								"authors": authors,
								"journal_ref": journal_ref,
								"updated": updated,
								"categories": categories,
								"real_category": category,
								"links": links,
								"fingerprint": category + '_' + str(pos)
						}
						
						documentList.append(document)

				start[clusterCategories[categoryId]] += batchSize[d] # get next n articles for the same category

		documents.insert_many(documentList)

client.close()