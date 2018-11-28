import numpy as np 
from collections import Counter
class knn(object):
	#train_date: m x n
	#label : [m]
	def __init__(self,train_data,label,k=4):
		self.k = k
		self.train_data = train_data
		self.label = label
	
	def knn_distance(self,predict_data):
		k = self.k
		train_data = self.train_data
		label = self.label
		distance = np.sqrt(np.sum(np.square(train_data - predict_data),axis = 1))
		#Bubble Sort
		for i in range(len(distance)-1):
			for j in range(len(distance)-1-i):
				if(distance[j]>distance[j+1]):
					#swap data
					tmp = distance[j]
					distance[j] = distance[j+1]
					distance[j+1] = tmp
					#swap label
					tmp = label[j]
					label[j] = label[j+1]
					label[j+1] = tmp
		top_k_label = label[0:k-1]
		knn_result = Counter(top_k_label)
		knn_list = list(knn_result)
		#print(knn_result.keys())
		#print(knn_result.elements())
		return knn_list[0]

	def predict(self,predict_data):
		return self.knn_distance(predict_data)

def main():
	train_data = np.array([[1.0,2.3,23.5],[4,92,1.07],[3.2,2.2,1.2]],dtype = np.float)
	label = ['object1','object2','object3']
	KNN = knn(train_data,label)
	predict_data = np.array([3,2.2,25],dtype = np.float)
	result = KNN.predict(predict_data)
	print(result)

if __name__ == '__main__':
	main()


