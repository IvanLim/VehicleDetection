from operator import itemgetter
import numpy as np
import cv2

# Using the very elegant Borg design pattern to maintain a shared state between instances
# Making it a much cleaner solution compared to a singleton
class Borg():
	_shared_state = {}
	def __init__(self): self.__dict__ = self._shared_state
	def __hash__(self): return 1
	def __eq__(self, other):
		try: return self.__dict__ is other.__dict__
		except: return 0

class InterFrameLogic(Borg):
	def __init__(self, first_init=False):
		Borg.__init__(self)
		if first_init:
			self.last_frame_heat_map = None
			self.previous_boxes = []
			self.classifier = None
			self.scaler = None

	# Smoothes out the boxes across frames
	# Bounding boxes are in the format ((x1, y1), (x2, y2))
	def average_boxes_across_frames(self, current_frame_boxes):
		# Reformat from ((x1, y1), (x2, y2)) to (x1, y1, x2, y2) for easier sorting/calculation
		reformatted_boxes = [(box[0][0], box[0][1], box[1][0], box[1][1]) for box in current_frame_boxes]
		self.previous_boxes.append(reformatted_boxes)
		previous_boxes = self.previous_boxes[-8:]

		if len(previous_boxes) == 1:
			final_boxes = reformatted_boxes

		# Sort the list of detected boxes
		flattened_boxes = []
		for boxes in previous_boxes:
			flattened_boxes.extend(boxes)

		clustered_boxes = self.cluster_boxes(flattened_boxes)

		# Only filter and drop items if we have enough of them
		if len(previous_boxes) >= 6:
			clustered_boxes = self.filter_clusters(clustered_boxes)

		averaged_clusters = self.average_clusters(clustered_boxes)

		final_boxes = [((box[0], box[1]), (box[2], box[3])) for box in averaged_clusters]

		return final_boxes

	# Clusters a given list of bounding boxes based on the amount of overlap
	# Bounding boxes are in the format [x1, y1, x2, y2]
	def cluster_boxes(self, boxes):
		# returns the amount of overlap between two rects. From 0 to 1
		def calc_overlap_percent(a, b):
			ax1, ay1, ax2, ay2 = a
			bx1, by1, bx2, by2 = b
				
			# If there is no overlap at all
			if (ax1 < bx1 and ax2 < bx1) or (ax1 > bx2 and ax2 > bx2) or \
				(ay1 < by1 and ay2 < by1) or (ay1 > by2 and ay2 > by2):
				return 0
			
			x1 = max(ax1, bx1)
			x2 = min(ax2, bx2)
			y1 = max(ay1, by1)
			y2 = min(ay2, by2)

			w = x2 - x1
			h = y2 - y1

			area_overlap = np.float64(np.abs(w) * np.abs(h))
			area_a = np.float64(np.abs(ax2 - ax1) * np.abs(ay2 - ay1))
			area_b = np.float64(np.abs(bx2 - bx1) * np.abs(by2 - by1))
			total_area = np.float64(area_a + area_b - area_overlap)

			overlap_percent = np.float64(area_overlap / total_area)
			return overlap_percent

		clusters = []
		min_overlap_percent = 0.1

		# For each box
		for box in boxes:
			# If no clusters yet, just create one
			if len(clusters) == 0:
				clusters.append([box])
			else:
				# look for existing clusters and calculate distances
				overlap_with_clusters = []
				for cluster in clusters:

					cluster.sort(key=itemgetter(0, 2))
					middle_index = len(cluster) // 2

					overlap = calc_overlap_percent(box, cluster[middle_index])
					overlap_with_clusters.append(overlap)

				biggest_overlap_index = np.argmax(overlap_with_clusters)

				# If we're within an acceptable overlap, consider it part of this cluster
				if (overlap_with_clusters[biggest_overlap_index]) >= min_overlap_percent:
					clusters[biggest_overlap_index].append(box)
				else:
				# it's an odd one out. Consider it a new cluster for now
					clusters.append([box])

		return clusters

	# If, after clustering across several frames, the cluster only has ONE box,
	# chances are it's a random spike/false positive. Drop it.
	def filter_clusters(self, clusters):
		filtered = []
		min_boxes_required = 3
		for cluster in clusters:
			if len(cluster) > min_boxes_required:
				filtered.append(cluster)
		return filtered

	def average_clusters(self, clusters):
		results = []
		for cluster in clusters:
			if len(cluster) == 1:
				results.append(np.array(cluster[0]))
			elif len(cluster) > 1:
				results.append(np.int64(np.around(np.mean(cluster, axis=0))))
		return results


################
# Unit testing #
################
# Run 'python InterFrameLogic.py' to test
if __name__ == '__main__':
	sc = InterFrameLogic(first_init=True)

# [[(800, 366, 959, 525), (800, 366, 959, 525), (800, 382, 959, 525), (800, 382, 959, 541)], [(1016, 398, 1183, 525), (1016, 398, 1199, 525), (1016, 398, 1199, 525), (1016, 398, 1199, 525)], [(816, 382, 975, 541)]]
	curr_boxes = [((800, 366),(959, 525)), ((800, 366), (959, 525)), ((1016, 398), (1183, 525)), ((816, 382), (975, 541))]
	sc.average_boxes_across_frames(curr_boxes)


	# curr_boxes = [((2,3),(4, 5)), ((11, 12), (13, 14))]
	# sc.average_boxes_across_frames(curr_boxes)

	# curr_boxes = [((3,4),(5,6)), ((12, 13), (14, 15))]
	# sc.average_boxes_across_frames(curr_boxes)

	# curr_boxes = [((3,4),(5,6)), ((16, 17), (18, 19))]
	# sc.average_boxes_across_frames(curr_boxes)

	# curr_boxes = [((5,6),(7,8)), ((15, 16), (17, 18))]
	# sc.average_boxes_across_frames(curr_boxes)

	# curr_boxes = [((6,7),(8,9)), ((150, 160), (170, 180))]
	# sc.average_boxes_across_frames(curr_boxes)

	# def center(box):			
	# 	x1, y1 = box[0]
	# 	x2, y2 = box[1]
	# 	return (((x2 - x1) / 2), ((y2 - y1) / 2))

	# def close_enough(box1, box2, distance_threshold):
	# 	center1 = center(box1)
	# 	center2 = center(box2)
	# 	dist = (center2[0] - center1[0], center2[1] - center1[1])
	# 	magnitude = np.sqrt(dist[0] ** 2 + dist[1] ** 2)

	# 	print('Mag: {}'.format(magnitude))

	# 	if magnitude <= distance_threshold:
	# 		return True
	# 	else:
	# 		return False

	# box1 = ((1068, 410), (1187, 505))
	# box2 = ((1008, 398), (1187, 505))
	# print(close_enough(box1, box2, 50))
	# sanity_checker.check_raw_bounding_boxes([1, 2, 3])

	# sc = InterFrameLogic()
	# sc.check_raw_bounding_boxes([4, 5, 6])

	# print(sanity_checker.cars)
	# print(sc.cars)