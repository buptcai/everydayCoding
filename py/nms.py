import numpy as np 
class nms(object):

	#input: (n,5)
	def __init__(self,bbx):
		self.bbx = bbx

	def non_maximum_suppression(self,iou_threshold=0.6):
		
		bbx = self.bbx
		bbx_x1 = bbx[:,0] #min
		bbx_y1 = bbx[:,1]
		bbx_x2 = bbx[:,2] #max
		bbx_y2 = bbx[:,3]
		bbx_score = bbx[:,4]
		bbx_area = (bbx_y2-bbx_y1)*(bbx_x2-bbx_x1)

		bbx_sort = np.argsort(bbx_score)[::-1] # from large to small
		final_bbx = []
		while bbx_sort.size>0:
			i = bbx_sort[0]
			final_bbx.append(bbx[i])

			#get iou
			ymin = np.maximum(bbx_y1[i],bbx_y1[bbx_sort[1:]])
			xmin = np.maximum(bbx_x1[i],bbx_x1[bbx_sort[1:]])
			ymax = np.minimum(bbx_y2[i],bbx_y2[bbx_sort[1:]])
			xmax = np.minimum(bbx_x2[i],bbx_x2[bbx_sort[1:]])
			h = np.maximum(0,ymax-ymin)
			w = np.maximum(0,xmax-xmin)
			inter_area = w*h #n-1
			iou = inter_area/(bbx_area[i]+bbx_area[bbx_sort[1:]]-inter_area)

			record = np.where(iou<=iou_threshold)[0] #np.where(condition) return a tuple
			bbx_sort = bbx_sort[record+1] #nb

		return final_bbx

	def soft_non_maximum_suppression(self,score_threshold=0.5):

		bbx = self.bbx
		bbx_x1 = bbx[:,0]
		bbx_y1 = bbx[:,1]
		bbx_x2 = bbx[:,2]
		bbx_y2 = bbx[:,3]
		bbx_score = bbx[:,4]

		bbx_sort = np.argsort(bbx_score)[::-1]
		bbx_area = (bbx_y2-bbx_y1)*(bbx_x2-bbx_x1) #2>1
		bbx_size = bbx_sort.size
		final_bbx = []
		pos = 0
		while pos<bbx_size-1:
			i = bbx_sort[pos]
			ymin = np.maximum(bbx_y1[i],bbx_y1[bbx_sort[pos+1:]])
			xmin = np.maximum(bbx_x1[i],bbx_x1[bbx_sort[pos+1:]])
			ymax = np.minimum(bbx_y2[i],bbx_y2[bbx_sort[pos+1:]])
			xmax = np.minimum(bbx_x2[i],bbx_x2[bbx_sort[pos+1:]])

			h = np.maximum(0,ymax-ymin)
			w = np.maximum(0,xmax-xmin)

			inter_area = w * h
			iou = inter_area/(bbx_area[i]+bbx_area[pos+1:]-inter_area)

			#linear
			weights = np.exp(-1*(np.square(iou))/0.5)
			bbx_score[bbx_sort[pos+1:]] = bbx_score[bbx_sort[pos+1:]]*weights #update score
			bbx_sort = np.argsort(bbx_score)[::-1] #update sort
			pos = pos + 1

		for j in range(bbx_size):
			if bbx_score[j]>score_threshold:
				final_bbx.append(bbx[j])

		return final_bbx


def main():
	bbox = np.array([[30, 20, 230, 200, 1], 
                     [50, 50, 260, 220, 0.9],
                     [40, 30, 256, 210, 0.8],
                     [430, 280, 460, 360, 0.7]])
	NMS = nms(bbox)
	final_box = NMS.non_maximum_suppression(iou_threshold=0.5)
	print(final_box)
	final_box_soft = NMS.soft_non_maximum_suppression(score_threshold=0.8)
	print(final_box_soft)

if __name__ == '__main__':
	main()




