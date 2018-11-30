import numpy as np 

class decisiontree(object):

	def __init__(self,origin_datasets,properties):
		#properties is a list
		self.origin_datasets = origin_datasets
		self.properties = properties
		self.length = np.array(properties).size

	def dtree_information_entropy(self,dataset):
		#dataset:m*(n+1) ,list or array 
		dataset = np.array(dataset)
		value = dataset[:,-1]
		value_size = value.size
		prob = []
		for i in range(np.unique(value).size):
			prob.append(np.sum(value==np.unique(value)[i])/value_size)
		Ent = -1*np.sum(prob*np.log2(prob))
		return Ent

	def dtree_information_gain(self,dataset,properties):
		#dataset:(m,n+1); properties:(n,); properties_v:(n,) property must be list
		gain = []
		dataset = np.array(dataset)
		Ent = self.dtree_information_entropy(dataset)
		for i in range(len(properties)):
			Ent_sum = 0
			properties_v = np.unique(dataset[:,j])

			for j in range(len(properties_v)):
				index_dataset_son_v = np.where(dataset[:,j]==properties_v[j])[0]
				dataset_son_v = dataset[index_dataset_son_v]
				weight = len(dataset_son_v)/len(dataset)
				Ent_son = self.dtree_information_entropy(dataset_son_v)
				Ent_sum = Ent_sum + weight*Ent_son

			gain_son = Ent-Ent_sum
			gain.append(gain_son)
		max_index = properties[gain.index(max(gain))]
		return max_index

	def dtree_treeGeneration(self,dataset,properties):
		dataset = np.array(dataset)
		value = dataset[:,-1]
		if(np.unique(value).size==1):
			#set it leaf point
			return
