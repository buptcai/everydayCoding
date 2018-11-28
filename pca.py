import numpy as np 

class pca(object):
	
	#data: array mxn ,m is number of sample,n is number of dim
	def __init__(self,data):
		self.data = data

	def pca_sort(self,eig,eig_vector,n):
		eig_index_sort = np.argsort(eig)
		eig_index_sort_trunc = eig_index_sort[:-(n+1):-1]
		eig_vector_trunc = eig_vector[:,eig_index_sort_trunc]
		return eig_vector_trunc

	def pca_predict(self,n):
		data = self.data
		feature_mean = np.mean(data,axis = 0)
		feature_centralization = data - feature_mean
		feature_C = np.cov(feature_centralization.T)
		eig,eig_vector = np.linalg.eig(feature_C)
		if n > eig_vector.shape[1]:
			return 'error'
		truncted_eig_vector = self.pca_sort(eig,eig_vector,n)
		feature_predict = np.dot(data,truncted_eig_vector)
		return feature_predict

def main():
	data = np.array([[10, 15, 29],
                        [15, 46, 13],
                        [23, 21, 30],
                        [11, 9,  35],
                        [42, 45, 11],
                        [9,  48, 5],
                        [11, 21, 14],
                        [8,  5,  15],
                        [11, 12, 21],
                        [21, 20, 25]])
	PCA = pca(data)
	feature_predict = PCA.pca_predict(2)
	print(feature_predict)

if __name__ == '__main__':
	main()

