import numpy as np 
class nms(object):

	#input: (n,5)
	def __init__(self,bbx,threshold):
		self.bbx = bbx
		self.threshold = threshold

	def nms(self):
		
		bbx = self.bbx
		bbx_x1 = bbx[:0] #min
		bbx_y1 = bbx[:1]
		bbx_x2 = bbx[:2] #max
		bbx_y2 = bbx[:3]
		bbx_score = bbx[:4]
		bbx_area = (bbx_y2-bbx_y1)*(bbx_x2-bbx_1)

		bbx_sort = np.argsort(bbx_score)[::-1] # from large to small
		final_bbx = []
		while condition:
			i = bbx_sort[0]
			final_bbx.append(bbx[i])

			#get iou
			ymin = np.maximum(bbx_y1[0],bbx_y1[bbx_sort[1:]])
			xmin = np.maximum(bbx_x1[0],bbx_x1[bbx_sort[1:]])
			ymax = np.minimum(bbx_y2[0],bbx_y2[bbx_sort[1:]])
			xmax = np.minimum(bbx_x2[0],bbx_x2[bbx_sort[1:]])
			w = np.maximum(0,ymax-ymin)
			h = np.maximum(0,xmax-xmin)
			inter_area = w*h #n-1
			iou = inter_area/(bbx_area[i]+bbx_area[bbx_sort[1:]]-inter_area)

			record = np.where(iou<=self.threshold)[0] #np.where(condition) return a tuple
			bbx_sort = bbx_sort[record+1] #nb



