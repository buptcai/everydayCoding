import numpy as np 

class decisiontree(object):

	def __init__(self,origin_datasets,properties):
		#properties is a list
		self.origin_datasets = origin_datasets
		self.properties = properties
		self.length = np.array(properties).size
		self.properties_copy = list(np.arange(self.length))
		self.tree = []

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
			properties_v = np.unique(dataset[:,properties[i]])
			for j in range(len(properties_v)):
				index_dataset_son_v = np.where(dataset[:,properties[i]]==properties_v[j])[0]
				dataset_son_v = dataset[index_dataset_son_v]
				weight = len(dataset_son_v)/len(dataset)
				Ent_son = self.dtree_information_entropy(dataset_son_v)
				Ent_sum = Ent_sum + weight*Ent_son

			gain_son = Ent-Ent_sum
			gain.append(gain_son)
		max_index = properties[gain.index(max(gain))]
		#print(properties)
		#print(max_index)
		#print(gain.index(max(gain)))
		return max_index,gain.index(max(gain))

	def dtree_treeGeneration(self,dataset,properties):
		dataset = np.array(dataset)
		value = dataset[:,-1]
		if(np.unique(value).size==1):
			#set it leaf point
			return

		dataset_pro = dataset[:,properties]
		if (len(properties)==0 or np.unique(dataset_pro,axis=0).shape[0]==1):
			#set it leaf point
			return
		best_gain_index,index = self.dtree_information_gain(dataset,properties)
		best_property = dataset[:,best_gain_index]
		property_value = np.unique(best_property)
		properties.remove(properties[index])
		for i in range(property_value.size):#the problem
			dataset_son = dataset[np.where(best_property==property_value[i])[0]]
			
			if dataset_son.size == 0:
				#
				return
			else:
				#properties have been changed
				self.tree.append([best_gain_index,property_value[i]])
				self.dtree_treeGeneration(dataset_son,properties)

		
trainData = [
        [0, 0, 0, 0,0],
        [0, 0, 0, 1,0],
        [0, 1, 0, 1,1],
        [0, 1, 1, 0,1],
        [0, 0, 0, 0,0],
        [1, 0, 0, 0,0],
        [1, 0, 0, 1,0],
        [1, 1, 1, 1,1],
        [1, 0, 1, 2,1],
        [1, 0, 1, 2,1],
        [2, 0, 1, 2,1],
        [2, 0, 1, 1,1],
        [2, 1, 0, 1,1],
        [2, 1, 0, 2,1],
        [2, 0, 0, 0,0],
    ]
properties=[0,1,2,3]
tree = decisiontree(trainData,properties)
tree.dtree_treeGeneration(trainData,properties)
print(tree.tree)






















