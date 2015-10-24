import Dimension as dim
# Generic class for vector handling in any dimension, with arbitrary name for each dimension
import math
import numpy as np

class Vector:

	name_ = 'vector'
	dimensions_ = []
	nbrOfPoints = 0

	def __init__(self):
		self.name_ = 'vector'
		self.dimensions_ = []
		self.nbrOfPoints = 0

	def SetName(self, name):
		self.name_ = name

	def GetName(self):
		return self.name_

	def GetNbrOfPoints(self):
		return self.nbrOfPoints

	def AddDimension(self, name):
		# print 'Adding dimension'
		new_dim = dim.Dimension(name)
		self.dimensions_.append(new_dim)
		
	def GetNbrOfDimensions(self):
		return len(self.dimensions_)

	def GetDimensions(self):
		return self.dimensions_

	def AddPoint(self, listOfValues):
		# Add values in the order dimensions were created. listOfValues is tuple
		# print 'listOfValues: ', listOfValues
		for i in range(len(listOfValues)):
			p = self.dimensions_[i].GetPoints()
			# print 'p: ', p
			# print 'lov[i]: ', listOfValues[i]
			# print 'i. ', i
			# print 'name: ', self.dimensions_[i].GetName()
			p.append(listOfValues[i])
		self.nbrOfPoints = self.nbrOfPoints +1
		# print 'NbrOfPoints is now: ', self.nbrOfPoints

	def SetPoint(self, dimension, index, newValue):
		for i in range(len(self.dimensions_)):
			if (self.dimensions_[i].GetName() == dimension):
				self.dimensions_[i].points_[index] = newValue

	def Max(self, dimension):
		# Compute maximum along the named dimension
		# print 'Vector_util.Max: ', dimension
		for i in range(len(self.dimensions_)):
			if (self.dimensions_[i].GetName() == dimension):
				maxValue = max(self.dimensions_[i].points_)
		return maxValue
	def Min(self, dimension):
		# print 'Vector_util.Min: ', dimension
		# Compute minimum along the named dimension
		# print 'Vector_util.Min'
		for i in range(len(self.dimensions_)):
			if (self.dimensions_[i].GetName() == dimension):
				# print 'Checking dimension: ', dimension
				minValue = min(self.dimensions_[i].points_)
				# print 'Inside, minimum: ', minValue
		return minValue
		

	def GetAllPointsWithValue(self, dimension, value):
		# Returns list of [index, coord1, coord2, ...]
		pointList = []
		for i in range(len(self.dimensions_)):
			if (self.dimensions_[i].GetName() == dimension):
				for p in range(len(self.dimensions_[i].points_)):
					if self.dimensions_[i].points_[p]  == value:
						newPoint = self.GetCompletePoint(p)
						pointList.append([p,newPoint])
		#print pointList
		return pointList

	def GetAllPoints(self):
		# Returns list of all points as [index, coord1, coord2, ...]
		pointList = []
		for i in range(len(self.dimensions_)):
			for p in range(len(self.dimensions_[i].points_)):
				newPoint = self.GetCompletePoint(p)
				pointList.append([p,newPoint])
		#print pointList
		return pointList

	def ComputeCurvatureGradient(self, index, scale, normalize):
		# Computes gradient in point at index (p) as -(p[scale] - p[-scale])
		# Normalize = 1 normalizes both vectors to unit before coputing gradient (no pixelerror)
		# Returns normalized (to length of scale) gradient as [x,y]
		# print 'Vector_util.ComputeCurvatureGradient'
		point = np.array(self.GetCompletePoint(index))
		print 'Point: ', point
		print 'Index: ', index
		print 'Scale: ', scale
		print 'Nbr of points: ', self.nbrOfPoints
		point_minus_scale = np.array(self.GetCompletePoint(np.mod(index - scale, self.nbrOfPoints)))
		# point_minus_scale = np.array(self.GetCompletePoint(index - scale))
		print 'Point-: ', point_minus_scale
		point_plus_scale = np.array(self.GetCompletePoint(np.mod(index + scale, self.nbrOfPoints)))
		# point_plus_scale = np.array(self.GetCompletePoint(index + scale))
		print 'Point+: ', point_plus_scale
		
		# v1 = point  - point_minus_scale
		v1 = point  - point_minus_scale
		# v2 = point_plus_scale - point
		v2 = point_plus_scale - point
		norm_to_1 = math.sqrt(1.0*v1[0]**2 + v1[1]**2) 
		norm_to_2 = math.sqrt(1.0*v2[0]**2 + v2[1]**2) 
		# print 'Norm to: ', norm_to
		if normalize > 0:
			v1 = 1.0 * norm_to_2 * v1/(norm_to_1)
			v2 = 1.0 * v2
		print 'v1: ', v1
		print 'v2: ', v2
		gradient = -1.0 * (v2 - v1)
		length_gradient = math.sqrt(gradient[0]**2 + gradient[1]**2)
		print 'Gradient: ', gradient
		gradient = (gradient / length_gradient) * 1 # 1 can be replaced with scale to get vector with length
		print 'Gradient: ', gradient
		return gradient
						
	def GetCompletePoint(self, index):
		point = []
		for i in range(len(self.dimensions_)):
			posToAppend = self.dimensions_[i].points_[index]
			point.append(posToAppend)
		return point

	def GetPoint(self, dimension, index):
		for i in range(len(self.dimensions_)):
			if self.dimensions_[i].GetName() == dimension:
				return self.dimensions_[i].points_[index]
		return 'NULL'

	def ComputeDistance(self, points1, points2):
		# Computes euclidean distance between two vectors, points1, points2
		# Returns double distance
		distance = 0.0
		points1 = np.array(points1)
		points2 = np.array(points2)
		distance = np.sqrt(sum((points1-points2)**2))
		return distance


	def GetPoints_1Dim(self, dimension):
		# Return all points along the named dimension
		# print 'Dimension: ', dimension
		for i in range(len(self.dimensions_)):
			# print self.dimensions_[i].GetName()
			if (self.dimensions_[i].GetName() == dimension):
				points = self.dimensions_[i].points_
				# print 'Returning: ', points
		return points

	def GetPoints_nDim(self, dimensions):
		# Return all points along the named dimensions (tuple)
		# print 'Dimension: ', dimension
		# print 'Vector_util.GetPoints_nDim', dimensions
		# print 'Points x: ', self.GetPoints_1Dim('x')
		points = []
		for i in range(self.nbrOfPoints):
			# print self.dimensions_[i].GetName()
			# print 'i: ', i
			points_per_i = []
			for dimIn in range(len(dimensions)):
				# print 'dimIn: ', dimIn
				for dimSelf in range(len(self.dimensions_)):
					# print 'dimSelf: ', dimSelf
					if (self.dimensions_[dimSelf].GetName() == dimensions[dimIn]):
						# print 'Added 1 point: ', self.dimensions_[dimSelf].points_[i], ' to ', self.dimensions_[dimSelf].GetName()
						points_per_i.append(self.dimensions_[dimSelf].points_[i])
			# print 'Added n points: ', points_per_i
			# print 'Points: ', points
			points.append(points_per_i)
				# print 'Returning: ', points
		return points

	def GetIndex(self, dimension):
		# Get the dimensional index for a dimension of name [dimension]
		for i in range(len(self.dimensions_)):
			if (self.dimensions_[i].GetName() == dimension):
				return i
		return 'NULL'

	def IsStraight(self, dimension1, dimension2):
		# Check if this string of points constitutes a straight line along named dimensions
		# Return 1 if yes
		# print 'Vector_util.IsStraight'
		# print 'vectorX: ', self.dimensions_[0].points_
		# print 'vectorY: ', self.dimensions_[1].points_
		index1 = self.GetIndex(dimension1)
		index2 = self.GetIndex(dimension2)
		
		firstPointX = self.dimensions_[index1].points_[0]
		firstPointY = self.dimensions_[index2].points_[0]
		# print 'firstX: ', firstPointX
		# print 'firstY: ', firstPointY
		lastPointX = self.dimensions_[index1].points_[self.nbrOfPoints-1]
		lastPointY = self.dimensions_[index2].points_[self.nbrOfPoints-1]
		# print 'lastX: ', lastPointX
		# print 'lastY: ', lastPointY
		deltaX = abs(lastPointX - firstPointX)
		deltaY = abs(lastPointY - firstPointY)
		# print 'deltaX: ', deltaX
		# print 'deltaY: ', deltaY
		# print 'NbrOfPoints: ', self.nbrOfPoints
		if deltaY == 0 and deltaX == self.nbrOfPoints -1:
			return 1
		if deltaX == 0 and deltaY == self.nbrOfPoints - 1:
			return 1
		if deltaX == deltaY and deltaX == self.nbrOfPoints -1: #Suspisious constructions
			return 1
		return 0


	def Is8Connected(self, dimension1, value1, dimension2, value2):
		# Check for 8connectivity with all other points in self for value1 in dimension1 and value2 in dimension2
		# Return 0 for false, 1 if true
		# print 'Vector_util.Is8Connected: [', dimension1, ':', value1, ',', dimension2, ':', value2, ']'
		index1 = self.GetIndex(dimension1)
		index2 = self.GetIndex(dimension2)
		for i in range(self.nbrOfPoints):
			# print 'Now checking against[', self.dimensions_[0].points_[i], ',', self.dimensions_[1].points_[i], ']'
			matchPoint = 0
			if (value1 == self.dimensions_[index1].points_[i]) and (value2 == self.dimensions_[index2].points_[i]):
				matchPoint = 1 #Same point
			elif (value1 == self.dimensions_[index1].points_[i] + 1 ) and (value2 == self.dimensions_[index2].points_[i]):
				matchPoint = 1 #west
			elif (value1 == self.dimensions_[index1].points_[i] - 1 ) and (value2 == self.dimensions_[index2].points_[i]):
				matchPoint = 1 #east
			elif (value1 == self.dimensions_[index1].points_[i] + 1 ) and (value2 == self.dimensions_[index2].points_[i] + 1):
				matchPoint = 1 #nw
			elif (value1 == self.dimensions_[index1].points_[i] + 1 ) and (value2 == self.dimensions_[index2].points_[i] - 1):
				matchPoint = 1 #ne
			elif (value1 == self.dimensions_[index1].points_[i] - 1 ) and (value2 == self.dimensions_[index2].points_[i] + 1):
				matchPoint = 1 #sw
			elif (value1 == self.dimensions_[index1].points_[i] - 1 ) and (value2 == self.dimensions_[index2].points_[i] - 1):
				matchPoint = 1 #se
			elif (value1 == self.dimensions_[index1].points_[i]) and (value2 == self.dimensions_[index2].points_[i]+ 1):
				matchPoint = 1 #north
			elif (value1 == self.dimensions_[index1].points_[i]) and (value2 == self.dimensions_[index2].points_[i] - 1):
				matchPoint = 1 #south
			if (matchPoint == 1):
				# print '8-connectivity found x: ', self.dimensions_[0].points_[i]
				# print '8-connectivity found y: ', self.dimensions_[1].points_[i]
				return 1

		return 0

	def IsInside(self, dimension1, value1, dimension2, value2):
		# Computes wether a point [value1, value2] is between the edges bound by the vector in dimension 1 and 2. 
		# Required is that at least along one dimension the value lies between two values in vector (e.g. betwwen 1 and 2 or between 3 and 4).
		# Requires point describing a continous curve, but not necesarily a closed curve
		# OBS funny things can happen for open structures
		# Parameters in: dimensions, value are tuples - both must have the same length
		# Returns 1 if inside, 0 as default
		points1 = self.GetPoints_1Dim(dimension1)
		points2 = self.GetPoints_1Dim(dimension2)
		# print 'value1: ', value1
		# print 'value2: ', value2
		# print 'points1: ', points1
		# print 'points2: ', points2

		# print 'Checking dim 1:'
		nbrOfPointsBelowOfValue = 0
		nbrOfPointsAboveOfValue = 0
		prevPoint2 = 0
		prevPoint1 = 0
		for i in range(len(points1)):
			if points1[i] == value1:
				# print 'Found equal point 1: ', value1
				point2 = self.GetPoint(dimension2, i)
				# print 'point2: ', point2
				# Check if point is above or below, plus do not allow sequence with former point
				if (point2 < value2) and (prevPoint2 +1 <> point2) and (prevPoint2 - 1 <> point2):
					nbrOfPointsBelowOfValue += 1
					prevPoint2 = point2
					# print 'nbrOfPointsBelowOfValue: ', nbrOfPointsBelowOfValue
				elif point2 > value2 and (prevPoint2 +1 <> point2) and (prevPoint2 - 1 <> point2):
					nbrOfPointsAboveOfValue += 1
					prevPoint2 = point2
					# print 'nbrOfPointsBelowOfValue: ', nbrOfPointsBelowOfValue
				elif point2 == value2 and (prevPoint2 +1 <> point2) and (prevPoint2 - 1 <> point2):
					# print 'nbrOfPointsBelowOfValue: Returning edge'
					return 1 # We are on edge

		# Check for dim1:
		if np.mod(nbrOfPointsAboveOfValue,2) == 1 and np.mod(nbrOfPointsBelowOfValue, 2) == 1:
			# We have an odd number of points above and below - point is likely inside.
			# print 'nbrOfPointsBelowOfValue: Returning 1'
			return 1

		# If no match for dim1: check for dim2
		# print 'Checking dim 2:'
		nbrOfPointsBelowOfValue = 0
		nbrOfPointsAboveOfValue = 0
		for i in range(len(points2)):
			if points2[i] == value2:
				# print 'Found equal point 2: ', value2
				point1 = self.GetPoint(dimension1, i)
				# print 'point1: ', point1
				# Check if point is above or below, plus do not allow sequence with former point
				if (point1 < value1)  and (prevPoint1 +1 <> point1) and (prevPoint1 - 1 <> point1):
					nbrOfPointsBelowOfValue += 1
					prevPoint1 = point1
					# print 'nbrOfPointsBelowOfValue: ', nbrOfPointsBelowOfValue
				elif (point1 > value1)  and (prevPoint1 +1 <> point1) and (prevPoint1 - 1 <> point1):
					nbrOfPointsAboveOfValue += 1
					prevPoint1 = point1
					# print 'nbrOfPointsBelowOfValue: ', nbrOfPointsBelowOfValue
				elif (point1 == value1)  and (prevPoint1 +1 <> point1) and (prevPoint1 - 1 <> point1):
					# print 'nbrOfPointsBelowOfValue: Returning edge'
					return 1 # We are on edge

		# Check for dim2:
		if np.mod(nbrOfPointsAboveOfValue,2) == 1 and np.mod(nbrOfPointsBelowOfValue, 2) == 1:
			# We have an odd number of points above and below - point is likely inside.
			# print 'nbrOfPointsBelowOfValue: Returning 1'
			return 1

		return 0

		# Compute line intersection
	def ComputeIntersection(self, k_m0, k_m1):
		# Returns [x,y] of intersection
		#print 'input0: ', k_m0
		#print 'input1: ', k_m1
		if k_m0[0] == 'INF' and k_m1[0] == 'INF':
			return 'NULL'
		elif k_m0[0] == 'INF':
			x = k_m0[1]
			y = round(k_m1[0]*x + k_m1[1])
		elif k_m1[0] == 'INF':
			x = k_m1[1]
			y = round(k_m0[0]*x + k_m0[1])
		else:
			k_ = np.array(k_m0)-np.array(k_m1)
			#print 'k_: ', k_
			if k_[0] <> 0:
				x = round((-1)*k_[1]/k_[0])
				y = round(k_m0[0]*x + k_m0[1])
		#		print 'return: [', x, ',', y, ']'
			else:
				return 'NULL' # Lines are parallell
		return [x,y]


