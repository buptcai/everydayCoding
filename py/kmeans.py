import numpy as np 

class kmeans(object):

	def __init__(self,k = 3):
		self.k = k

	def kmeans_predict(self,data,iter):
		#initialize centers
		k = self.k
		center = np.zeros((k,data.shape[1]),dtype = np.float)
		for i in range(k):
			center[i] = data[i]
		distance = self.kmeans_distance(center,data)#distance = (k,n)
		label = self.kmeans_label(distance) #label = (n,)
		#Loop
		for i in range(iter):
			distance = self.kmeans_distance(center,data)
			label = self.kmeans_label(distance)
			center = self.kmeans_update(data,label,center)

		final_distance = self.kmeans_distance(center,data)
		final_label = self.kmeans_label(final_distance)
		final_center = self.kmeans_update(data,final_label,center)

		return final_center,final_label

	def kmeans_distance(self,center,data):
		k = self.k
		#initialize distance
		distance =np.zeros((k,data.shape[0]),dtype = np.float)
		for i in range(k):
			distance[i] = np.sqrt(np.sum(np.square(data-center[i]),axis = 1))
		return distance

	def kmeans_label(self,distance):
		label = distance.argmin(axis = 0) #(n,)
		return label

	def kmeans_update(self,data,label,center):
		k = self.k
		new_center = np.zeros((k,data.shape[1]),dtype = np.float)
		count = np.zeros(k)
		for j in range(data.shape[0]):
			for i in range(k):
				if(label[j] == i):
					new_center[i] = new_center[i] + data[j]
					count[i] = count[i] + 1
		for m in range(k):
			new_center[m] = new_center[m]/count[m]
		delta_center = np.sqrt(np.sum(np.square(center-new_center),axis = 1))
		return new_center		#(k,n)

	
def main():
	data = np.random.randint(0,100,(100,5))
	Kmeans = kmeans(k=6)
	center,label = Kmeans.kmeans_predict(data,10)
	print(center)
	print(label)

if __name__ =='__main__':
	main()




