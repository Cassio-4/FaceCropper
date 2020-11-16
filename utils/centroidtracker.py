from scipy.spatial import distance as dist
from utils.common import TrackableObject
import numpy as np


def deregister(obj_id, objects):
	# to deregister a TrackableObject, we delete it from the dict
	# and return the object and where it disappeared to be processed later
	return objects.pop(obj_id)


def calc_centroid(x, y, x1, y1):
	cx = int((x + x1) / 2.0)
	cy = int((y + y1) / 2.0)
	return cx, cy


class CentroidTracker:
	"""
	This class is the bread and butter of this program.
	That said, it is a mess and should be refactored asap!
	It does the ID and bounding boxes assignment.
	"""
	def __init__(self, max_disappeared=50, max_distance=50):
		"""
		Args:
			max_disappeared (int): Maximum number of frames that an object can
				remain disappeared before it is deleted.
			max_distance (int): The maximum distance between centroids before
				we start to mark them as disappeared.
		"""
		# Store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = max_disappeared

		# store the maximum distance between centroids to associate
		# an object
		self.maxDistance = max_distance

		self.next_id = 0

	def reset_id_count(self):
		self.next_id = 0

	def register(self, bbox, objects, face_crop=None, frame=None, score=None):
		"""
		Arguments:
			bbox: ndarray uint16 with bounding box info
				[left, top, right, bottom].
			objects: OrderedDict of ids and TrackableObjects.
			face_crop:
			score:
			frame:
		"""
		obj = TrackableObject(self.next_id, np.asarray(bbox, dtype=np.uint16))
		# Save the centroid pos when the object was registered
		obj.centroid_when_registered = obj.centroid
		# Save the new TrackableObject in the dictionary
		objects[self.next_id] = obj
		# Update the next Id
		self.next_id += 1
		# If the b.box comes from a detector, then there is a score and a
		# cropped image associated to it
		if (face_crop is not None) and (score is not None):
			obj.update_highest_detection(face_crop=face_crop, frame=frame, score=score)

	def no_bbox_input(self, objects):
		# loop over any existing tracked objects and mark them
		# as disappeared
		deregistered_objects = []
		for obj_id in list(objects.keys()):
			objects[obj_id].disappeared_frames += 1

			# if we have reached a maximum number of consecutive
			# frames where a given object has been marked as
			# missing, deregister it
			if objects[obj_id].disappeared_frames > self.maxDisappeared:
				deregistered_objects.append(deregister(obj_id, objects))
		return deregistered_objects

	def update(self, rects, objects, images=None, frame=None, scores=None):
		"""
		:param frame:
		:param rects: bounding boxes [[x_left, y_top, x_right, y_bottom], ...]
		:param objects: OrderedDictionary of id:TrackableObject.
		:param images: List of cropped images from frame.
		:param scores: Scores of detections in images argument.
		:return: modifies the objects associating boundingboxes,
			centroids and Ids. Also returns a list of no longer active objects
		"""
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# return early as there are no centroids or tracking info
			# to update
			return self.no_bbox_input(objects)

		# if we are currently not tracking any objects take the input
		# b.boxes and register each of them
		if len(objects) == 0:
			# If there are no scores and detections
			if (images is None) and (scores is None):
				for i in range(0, len(rects)):
					self.register(rects[i], objects)
			# Else, the rects are coming from a detector, with cropped faces
			# and scores to each cropping
			else:
				for i in range(0, len(rects)):
					self.register(rects[i], objects, face_crop=images[i], frame=frame, score=scores[i])
			# Return an empty list since all objects are fresh
			return []

		# otherwise, we are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# initialize an array of input centroids for the current frame
			input_centroids = np.zeros((len(rects), 2), dtype=np.uint16)
			# loop over the bounding box rectangles
			for (i, (startX, startY, endX, endY)) in enumerate(rects):
				# use the bounding box coordinates to derive the centroid
				input_centroids[i] = calc_centroid(startX, startY, endX, endY)

			# grab the set of object IDs and corresponding centroids
			object_ids = list(objects.keys())
			object_centroids = [objects[obj_id].centroid for obj_id in object_ids]

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			distances = dist.cdist(np.array(object_centroids), input_centroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = distances.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = distances.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			used_rows = set()
			used_cols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in used_rows or col in used_cols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if distances[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new bounding box, and reset the disappeared
				# counter
				object_id = object_ids[row]
				objects[object_id].set_bounding_box(rects[col])
				objects[object_id].disappeared_frames = 0
				# If the b.boxes came from a detector then associate the
				# cropped image and score to this object as well
				if images is not None and images[col] is not None:
					objects[object_id].update_highest_detection(images[col], frame, scores[col])

				# Indicate that we have examined each of the row and
				# column indexes, respectively
				used_rows.add(row)
				used_cols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
			unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			deregistered_objects = []
			if distances.shape[0] >= distances.shape[1]:
				# loop over the unused row indexes
				for row in unused_rows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					object_id = object_ids[row]
					objects[object_id].disappeared_frames += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if objects[object_id].disappeared_frames > self.maxDisappeared:
						deregistered_objects.append(deregister(object_id, objects))
				# This is a gambiarra I have no idea why it's here but it fixed
				# the None type obejct being created lol
				if images is not None:
					for col in unused_cols:
						self.register(rects[col], objects, images[col], frame, scores[col])

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unused_cols:
					if images is not None:
						self.register(rects[col], objects, images[col], frame, scores[col])
					else:
						self.register(rects[col], objects)

			return deregistered_objects
