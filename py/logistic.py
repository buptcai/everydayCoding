import numpy as np 

class logisticRegression(object):

	#input: train:(m,n);label:(m,) including +1 and -1
	def __init__(self,train,label,learning_rate=0.01,step=100):
		self.train = train
		self.label = label
		self.learning_rate = learning_rate
		self.step = step

	def logistic_train(self):
		#pre data
		m,n = self.train.shape
		self.m = m
		self.n = n
		train = np.ones((m,n+1),dtype = np.float)
		train[:,1:] = self.train
		#pre weights
		weights = np.ones(n+1,dtype = np.float)

		#train
		#update
		for j in range(self.step):
			for i in range(m):
				weights = weights + self.learning_rate*(self.label[i]-self.logistic_sigmoid(train[i],weights))*train[i]
		self.weights = weights
		print(self.weights)
	
	def logistic_sigmoid(self,data,weights):
		#sigmoid:1/(e^(-x)+1)
		hx = 1/(np.exp(-1*(np.sum(data*weights)))+1)
		return hx

	def predict(self,test):
		test = np.reshape(test,(-1,self.n))
		fx = np.sum(test*self.weights[1:],axis = 1)+self.weights[0]
		hx = 1/(np.exp(-1*fx)+1)
		index = np.where(hx<0.5)
		result = np.ones(test.shape[0])
		result[index] = 0
		return result

def main():
	train_data = np.array([[-0.017612,14.053064],[-1.395634,4.662541],[-0.752157,6.538620],
						   [-1.322371,7.152853],[0.423363,11.054677],[0.406704,7.067335],
						   [0.667394,12.741452],[-2.460150,6.866805],[0.569411,9.548755],
						   [-0.026632,10.427743]])
	label = np.array([0,1,0,0,0,1,0,1,0,0])
	LR = logisticRegression(train_data,label)
	LR.logistic_train()
	print(LR.predict(train_data))

if __name__ == '__main__':
	main()