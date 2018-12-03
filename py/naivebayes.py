import numpy as np

class naivebayesclassifier(object):

	#data:(m,n),label:(m,) list
	 def __init__(self,train_data,train_label):
	 	self.data = np.array(train_data)
	 	self.label = np.array(train_label)

	 def naivebayes_predict(self,predict_data):
	 	category = np.unique(self.label)
	 	N = category.size
	 	final_prob = []
	 	for i in range(N):

	 		Dc = np.sum(self.label==category[i])
	 		prior_prob = (Dc+1)/(self.label.size+N)
	 		prob = prior_prob
	 		#feature
	 		for j in range(self.data.shape[1]):
	 			feature = np.unique(self.data[:,j])
	 			Ni = feature.size #Ni
	 			condition_prob = (np.sum(self.data[:,j]==predict_data[j])+1)/(Dc+Ni)
	 			prob = prob*condition_prob

	 		final_prob.append(prob)
	 	return final_prob

def main():
	#waiting for validation

if __name__ == '__main__':
	main()