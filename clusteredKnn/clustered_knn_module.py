import numpy as np
import pandas
import os
from sklearn import neighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity

class ClusteredKnnModule(object):
	data_set_samples = []
	data_set_labels = []
	test_data_set_samples = []
	test_data_set_labels = []
	distance = []
	clusters = []
	clusters_training = []
	indices = []
	k_neighbors = 1
	n_clusters = 2

	def __init__(self):


	def run(self):




		#exit()
		'''self.findClusters(self.data_set_samples)

		self.findNearestNeighbors(self.data_set_samples)


		#exit()



		clf = neighbors.KNeighborsClassifier(self.k_neighbors,  weights='distance')
		clf.fit(self.distance, self.data_set_labels)



		self.findClusters(self.test_data_set_samples, self.data_set_samples)
		self.findNearestNeighbors(self.test_data_set_samples)

		
		predictions = clf.predict(self.distance)




		#exit()
		self.findClustersTraining(self.data_set_samples)
		self.findNearestNeighborsTraining(self.data_set_samples)



		clf = neighbors.KNeighborsClassifier(self.k_neighbors,  weights='distance')
		clf.fit(self.distance, self.data_set_labels)


		self.findClustersTesting(self.test_data_set_samples, self.data_set_samples)
		self.findNearestNeighborsTesting(self.test_data_set_samples)


		exit()
		predictions = clf.predict(self.distance)

		#exit()
		return predictions

	def findClustersTraining(self, data_set):
		self.distance = []
		self.clusters = []
		self.labels = []
		self.indices = []

		kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0).fit(data_set)
		distance_clusters = kmeans.fit_transform(data_set)

		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.indices.append([])

		for i in range(0,len(data_set)):
			dist = 0
			for j in distance_clusters[i]:

				dist+= j 

			self.distance.append([dist])
			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(data_set[i]) 

	def findNearestNeighborsTraining(self, data_set):
		for i in range(0, len(self.clusters)):
			if(len(self.clusters[i]) > 1):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				clf.fit(self.clusters[i])
				for j in  range(0, len(self.clusters[i])):

					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)


					self.distance[self.indices[i][j]]+= neighbor[0][0][1]






					#exit()
			else:


	def findClustersTesting(self, test_data_set, data_set_training):
		self.distance = []
		self.clusters = []
		self.labels = []
		self.indices = []
		self.clusters_training = []

		mergedlist = []
		mergedlist.extend(test_data_set)
		mergedlist.extend(data_set_training)
		data_set_training = mergedlist

		kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0).fit(data_set_training)
		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.indices.append([])
			self.clusters_training.append([])





		for i in range(0, len(data_set_training)):
			self.clusters_training[kmeans.labels_[i]].append(data_set_training[i]) 






		#exit()

		distance_clusters = kmeans.fit_transform(test_data_set)



		#exit()
		for i in range(0,len(test_data_set)):
			dist = 0
			for j in distance_clusters[i]:

				dist+= j 

			self.distance.append([dist])
			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(test_data_set[i]) 

	def findNearestNeighborsTesting(self, test_data_set):
		for i in range(0, len(self.clusters_training)):
		
			if(len(self.clusters_training[i]) > 1):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				clf.fit(self.clusters_training[i])
				for j in  range(0, len(self.clusters[i])):

					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)


					self.distance[self.indices[i][j]]+= neighbor[0][0][1]






			else:

				
				#exit()
				#exit()
	'''def findClusters(self, data_set,  data_set_training=None):
		self.clusters = []
		self.clusters_training = []
		self.indices = []
		self.distance = []
		self.labels = []

		if (data_set_training == None):
			data_set_training = data_set
		else:
			mergedlist = []
			mergedlist.extend(data_set)
			mergedlist.extend(data_set_training)
			data_set_training = mergedlist




		#exit()

		kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0).fit(data_set_training)
		#predicao = kmeans.predict(data_set_training)



		#exit()

		distance_clusters = kmeans.fit_transform(data_set_training)




		#distance_clusters_data_set = distance_clusters[0:len(data_set)]

		#exit()


		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.clusters_training.append([])
			self.indices.append([])

		for i in range(0, len(data_set_training)):
			self.clusters_training[kmeans.labels_[i]].append(data_set_training[i]) 





		#exit()
		predicao = kmeans.predict(data_set_training)

		for i in range(0,len(data_set)):
			dist = 0

			for j in distance_clusters[i]:

				dist+= j 



			#self.distance.append(dist)
			self.distance.append([dist])






			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(data_set[i]) 


			#exit()
		#exit()



		return distance_clusters

	def findNearestNeighbors(self, data_set):



		#exit()

		for i in range(0, len(self.clusters)):
			if(len(self.clusters_training[i]) > 1):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				clf.fit(self.clusters_training[i])
				for j in  range(0, len(self.clusters[i])):

					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					self.distance[self.indices[i][j]]+= neighbor[0][0][1]






			else:
				clf = neighbors.NearestNeighbors(n_neighbors=1)
				clf.fit(self.clusters_training[i])
				for j in  range(0, len(self.clusters[i])):

					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					self.distance[self.indices[i][j]]+= neighbor[0][0][0]






		#exit()			

	'''
	def setDataSetClustering(self, data_set):
		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-1)]


	def setDataSet(self, data_set):

		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-2)]
		self.data_set_labels = data_set.values[:,(len(data_set.values[0])-2)]

		#exit()




	
	def setTestDataSet(self, test_data_set):



		self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		#self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		self.test_data_set_labels = test_data_set.values[:,(len(test_data_set.values[0])-2)]		





		#exit()
	

	def setKNeighbors(self, k_neighbors):
		self.k_neighbors = k_neighbors

	def getKNeighbors(self):
		return self.k_neighbors

	def setNClusters(self, n_clusters):
		self.n_clusters = n_clusters

	def getNClusters(self):
		return self.n_clusters

	

	'''def findNearestNeighbors(self, data_set, labels):



	 	for i in range(0, len(self.clusters)):
	 		clf = neighbors.KNeighborsClassifier()
			#clf = neighbors.NearestNeighbors(n_neighbors=2)
			clf.fit(self.clusters_training[i], self.labels[i])
			a = clf.kneighbors(n_neighbors=1)


			exit()


			#clf.fit(self.clusters_training[i])
			for j in  range(0, len(self.clusters[i])):





				#neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)


				#self.distance[self.indices[i][j]]+= neighbor[0][0][1]
		exit()



	def findClusters(self, data_set, data_set_labels,  data_set_training=None):
		self.clusters = []
		self.clusters_training = []
		self.indices = []
		self.distance = []
		self.labels = []
		
		if (data_set_training == None):
			data_set_training = data_set
		else:
			mergedlist = []
			mergedlist.extend(data_set)
			mergedlist.extend(data_set_training)
			data_set_training = mergedlist


		kmeans = MiniBatchKMeans(n_clusters=2, random_state=0).fit(data_set_training)
		#predicao = kmeans.predict(data_set_training)



		distance_clusters = kmeans.fit_transform(data_set_training)



		#distance_clusters_data_set = distance_clusters[0:len(data_set)]



		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.labels.append([])
			self.clusters_training.append([])
			self.indices.append([])

		for i in range(0, len(data_set_training)):
			self.clusters_training[kmeans.labels_[i]].append(data_set_training[i]) 




		predicao = kmeans.predict(data_set_training)

		for i in range(0,len(data_set)):
			dist = 0
			for j in distance_clusters[i]:
				dist+= j 

			#self.distance.append(dist)
			self.distance.append([dist])




			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(data_set[i]) 
			self.labels[kmeans.labels_[i]].append(data_set_labels[i]) 

		#exit()



		return distance_clusters

	def findNearestNeighbors(self, data_set, labels=None):




		if(labels == None):
			for i in range(0, len(self.clusters)):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				for j in  range(0, len(self.clusters[i])):
					clf.fit(self.clusters_training[i])
					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)

					self.distance[self.indices[i][j]]+= neighbor[0][0][1]
		else:
		 	for i in range(0, len(self.clusters)):
		 		clf = neighbors.KNeighborsClassifier()
			
				clf.fit(self.clusters_training[i], self.labels[i])
				neighbor = clf.kneighbors(n_neighbors=1)

				#exit()


				#clf.fit(self.clusters_training[i])
				for j in  range(0, len(self.clusters[i])):





					#neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)



					#exit()
					self.distance[self.indices[i][j]]+= neighbor[0][j]
		#exit()


	'''