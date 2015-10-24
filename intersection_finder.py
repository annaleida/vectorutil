import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage.filters import gaussian_laplace
from scipy.misc import imfilter
import numpy.random as rnd
import Vector_util as vut

class IntersectionFinder:


	def __init__(self):
		print 'IntersectionFinder initiated'


	# Get the index from a certain point in vector v with match on x and y
	def GetPointIndex(self, v, x, y):
		print '***GetPointIndex***'
		# print 'x: ', x
		# print 'y: ', y
		index = 0
		pointsX = v.GetAllPointsWithValue('x', x)
		# print 'Length x: ', len(pointsX)
		found = 0
		for i in range(len(pointsX)):
			if (pointsX[i][1][1] == y):
				pointsXY = pointsX[i]
				found = found + 1
				print 'GetPointIndex returning point', pointsXY
				index = pointsXY[0]
		# pointsXY = pointsX.GetAllPointsWithValue('y', y)
				if found > 1:
					print 'Warning: GetPointIndex returned more than one point.'
		return index


	def Run(self, a, pixelStep, plot_all_figures_):

		crop_ = 0 #Testing parameter for cutting images
		vectorComputations = vut.Vector()
		# Adds up to [input] zeros before integer
		pad_string = lambda integer : '0'*(nbrOfIndexSpaces-len(str(integer))) + str(integer)

		# OBS! Can only handle up to 1000 vectors with this setup
		maxNbrOfVectors = 1000
		nbrOfIndexSpaces = len(str(maxNbrOfVectors-1))
		maxNbrOfVectors = 10**nbrOfIndexSpaces
		print 'There is space for: ', maxNbrOfVectors, ' in this calculation'



		# a is numpy array in two dimensions


		edges = imfilter(a,'find_edges')
		edges = where(edges >=1, 1, 0)
		#Tracing---------------------------------
		d = nonzero(edges) #create tuple y = d[0,:], x = d[1,:]
		vec = [d[0][0],d[1][0]] #Pick first non-zero value as y,x
		traced_edge = zeros_like(edges) # Matrix to store values in image format
		# traced_edge[vec[0], vec[1]] = 1

		cropping = array([[100,200], [200,300]])
		edges2 = edges[cropping[0,0]:cropping[0,1], cropping[1,0]:cropping[1,1]]

		edges[vec[0], vec[1]] = -2 #Mark starting pixel
		continue_ = 1
		loop_ = 0
		size_ = 0
		previous_size_ = 0
		vec_outline = vut.Vector()
		vec_outline.AddDimension('x')
		vec_outline.AddDimension('y')
		vec_outline.SetName('out_' + pad_string(1))

		while continue_:
			#print 'this: ', [vec[0], vec[1]]
			loop_+=1
			north = [vec[0]+1,vec[1]]; #print 'edge(north): ', edges[north[0], north[1]];
			east = [vec[0], vec[1] +1]; #print 'edge(east): ', edges[east[0], east[1]];
			south = [vec[0]-1,vec[1]]; #print 'edge(south): ', edges[south[0], south[1]];
			west = [vec[0], vec[1] -1]; #print 'edge(west): ', edges[west[0], west[1]];
			ne = [vec[0]+1,vec[1]+1]
			se = [vec[0]-1, vec[1] +1]
			sw = [vec[0]-1,vec[1]-1]
			nw = [vec[0]+1, vec[1] -1]


			traced_edge[vec[0], vec[1]] = 1; #Add current point to stack
			# print 'vec B4: ', (vec[1], vec[0])
			vec_outline.AddPoint((vec[1], vec[0]))
			# break
			if edges[north[0], north[1]] >= 1:
				vec = north; edges[north[0], north[1]] = -1; #print 'north'; check_starting_point = 1;
			elif edges[east[0], east[1]] >= 1:
				vec = east; edges[east[0], east[1]] = -1; #print 'east'; check_starting_point = 1;
			elif edges[south[0], south[1]] >= 1:
				vec = south; edges[south[0], south[1]] = -1; #print 'south'; check_starting_point = 1;
			elif edges[west[0], west[1]] >= 1:
				vec = west; edges[west[0], west[1]] = -1; #print 'west'; check_starting_point = 1;
			elif edges[ne[0], ne[1]] >= 1:
				vec = ne; edges[ne[0], ne[1]] = -1; #print 'ne'; check_starting_point = 1;
			elif edges[se[0], se[1]] >= 1:
				vec = se; edges[se[0], se[1]] = -1; #print 'se'; check_starting_point = 1;
			elif edges[sw[0], sw[1]] >= 1:
				vec = sw; edges[sw[0], sw[1]] = -1; #print 'sw'; check_starting_point = 1;
			elif edges[nw[0], nw[1]] >= 1:
				vec = nw; edges[nw[0], nw[1]] = -1; #print 'nw'; check_starting_point = 1;

			size_ = vec_outline.GetNbrOfPoints()

			# Check if next position is on trace
			if traced_edge[vec[0], vec[1]] >= 1:
				print 'Found starting point! ', vec
				print '@loop:', loop_, ' the size (after tracing) is: ', size_
				continue_ = 0
			elif size_ == previous_size_:
				print 'Size if tracing vector is frozen at: ', vec
				print '@loop:', loop_, ' the size (after tracing) is: ', size_
				print 'Breaking loop'
				continue_ = 0

			previous_size_ = size_

		# vv = [vec, vec1]
		# for x in range(vec[:,0])
		# print vec 


		# b[d] = 1


		#Check for concavity -------------------------------------------------------
		print '***Check concavity***'
		# print 'x: ', vec_outline.GetPoints_1Dim('x')
		# print 'first dim: ', vec_outline.GetDimensions()[0].GetName()
		vec = array(vec_outline.GetPoints_nDim(('y', 'x')))
		# print 'test vec: ', vec
		# print 'vec2 test: ', vec2
		# print len(vec)
		mem = zeros_like(a).astype(float)
		step_length = pixelStep
		isConcavity = 0
		newConcavity = 0
		all_Concavities = [] #Vector to store all concavities as homemade vector class
		appendVector = 1
		# step_length = 25
		currentConcavity = vut.Vector() #Workvariable for current vector, to be added to all_Concavities. Set to null when new vector found
		currentConcavity.SetName('con_000') #Initialize will be overwritten
		for i in range(step_length,(len(vec) - step_length)):
			# print 'index: ', i
			# print 'vec: ', vec[i]
			# print 'vec+: ', vec[i+step_length]
			# print 'vec-: ', vec[i-step_length]
			v1 = array(vec[i+step_length] - vec[i])
			v2 = array(vec[i] - vec[i-step_length])
			# print v1
			# print v2
			# dot_ = abs(dot(v1,(-1)*v2))
			# print dot_
			if abs(cross(v1,-v2)) <> 0:
				sign_cross = cross(v1,-v2)/abs(cross(v1,-v2))
				# print sign_cross
				dot_product = abs(dot(v1,(-1)*v2))
				if sign_cross == -1:
					if isConcavity == 0: #Enter new concavity
						newConcavity = 1
						isConcavity = 1
					if newConcavity == 1:
						# Check for 8-connectivity with existing vectors
						# print len(all_Concavities)
						for v in range(len(all_Concavities)):
							# test = all_Concavities[v].Is8Connected([vec[i][1], vec[i][0]])
							if all_Concavities[v].Is8Connected('x', vec[i][1], 'y', vec[i][0]) == 1:
								currentConcavity = all_Concavities[v]
								# print 'Continuing vector: ', currentConcavity
								appendVector = 0
								newConcavity = 0
					if newConcavity == 1: #We have not found an existing vector to append to
						prev_name = currentConcavity.GetName()
						currentConcavity = vut.Vector()
						currentConcavity.SetName('con_' + pad_string(int(prev_name[-nbrOfIndexSpaces:]) + 1))
						currentConcavity.AddDimension('x')
						currentConcavity.AddDimension('y')
						currentConcavity.AddDimension('value')
						# currentConcavity.AddDimension('active') # To turn off paired concavities in filtering
						# print 'Found new concavity at [y,x]: ', vec[i]
						appendVector = 1
						newConcavity = 0
				
					mem[vec[i][0], vec[i][1]] = 1
					# mem[vec[i][0], vec[i][1]] = dot_product
					currentConcavity.AddPoint(([vec[i][1],vec[i][0],dot_product]))
					# print 'Adding point: x: ', [vec[i][1], ' y: ', vec[i][0], 'value: ', dot_product]
				else:
					if isConcavity == 1: #Exit concavity
						# print 'Exit concavity at: ', vec[i]
						if appendVector == 1: #Do not append if we are continuing an old vector
							all_Concavities.append(currentConcavity)
						isConcavity = 0

		# Filter out all straight and lonely segments will get rid of most false concavities--------------------

		print '***Filtering***'

		print 'Number of concavities found: ', len(all_Concavities)

		new_all_Concavities = []
		for i in range(len(all_Concavities)):
			if all_Concavities[i].GetNbrOfPoints() > 2 and all_Concavities[i].IsStraight('y', 'x') == 0:
				new_all_Concavities.append(all_Concavities[i])

		all_Concavities = new_all_Concavities
		print 'Number of concavities after filtering: ', len(all_Concavities)


		# for i in range(len(all_Concavities)):
			# print all_Concavities[i].GetName()


		#Rescale image -----------------------------------------------------------------

		print '***Rescaling***'

		mem2 = zeros_like(a).astype(float)
		for i in range(len(all_Concavities)):
			# print all_Concavities[i].GetPoints('value')
			maximum = all_Concavities[i].Max('value')
			# print 'Maximum: ', maximum
			# pointsX = all_Concavities[i].GetPoints_nDim(('x'))
			pointsXY = all_Concavities[i].GetPoints_nDim(('x', 'y'))
			# pointsYX = all_Concavities[i].GetPoints_nDim(('y', 'x'))
			# print 'Points x is: ', pointsX
			# print 'Points xy is: ', pointsXY
			# print 'Points yx is: ', pointsYX
			# pointsY = all_Concavities[i].GetPoints_1Dim('y')
			# pointsVal = all_Concavities[i].GetPoints_1Dim('value')

			# print 'initial value: ', all_Concavities[i].GetPoints('value')
			for p in range(len(pointsXY)):
				unNormalizedValue = float(all_Concavities[i].GetPoints_1Dim('value')[p])
				# print 'unNormalizedValue: ', unNormalizedValue
				normalizedValue = unNormalizedValue/maximum
				# print 'normalizedValue: ', normalizedValue
				all_Concavities[i].SetPoint('value', p, normalizedValue)
				mem2[pointsXY[p][1], pointsXY[p][0]] = normalizedValue
			# print 'After rescaling: ', all_Concavities[i].GetPoints_1Dim('value')
					
		# print 'Dim: ', mem.shape
		# print 'Max: ', mem.max()
		# print 'Min: ', mem.min()
			# mem = (mem/(mem.max()-mem.min())) + mem.min()
		# print 'Max: ', mem.max()
		# print 'Min: ', mem.min()


		# Mark minimum of each concavity (hopefully central point)---------------------------------
		print '***Find center***'

		corners = zeros_like(a).astype(float)
		corners = where(traced_edge == 1, 0.2, 0)
		vec_corners = []
		gradients = []
		for i in range(len(all_Concavities)):
			print all_Concavities[i].GetPoints_1Dim('value')
			minimum = all_Concavities[i].Min('value')
			# print 'Minimum: ', minimum
			#Select only center point
			findCenter_points = all_Concavities[i].GetAllPointsWithValue('value', minimum) 
			#Draw gradient from all points
			# findCenter_points = all_Concavities[i].GetAllPoints()
			
			# print 'Points: ', findCenter_points
			# print 'Length: ', len(findCenter_points) 
			# if len(findCenter_points) == all_Concavities[i].GetNbrOfPoints:
				# print 'All minimum'
			# Several minimum: sort out the middle one
			if len(findCenter_points) > 1:
				print 'Several minimum found: '
				total_mean = 0
				for h in range(len(findCenter_points)):
					total_mean = total_mean + findCenter_points[h][0]
					print findCenter_points[h][0]
				mean = round(total_mean/(len(findCenter_points)))
				diff_smallest = 100
				chosen_point = findCenter_points[0][0] # Choose first point as default
				for h in range(len(findCenter_points)):
					mean_diff = mean - findCenter_points[h][0]
					# print 'Mean diff: ', mean_diff
					# print 'Diff smallest: ', diff_smallest
					if abs(mean_diff) < abs(diff_smallest):
						diff_smallest = mean_diff
						chosen_point = findCenter_points[h]
				# chosen_point = uint8(mean - mean_diff)				
				print 'Middle minimum is: ', chosen_point
				findCenter_points = [chosen_point]
			print 'Middle minimum len is: ', len(findCenter_points)

			for j in range(len(findCenter_points)):
		# Mark gradient in these points
				myIndex = self.GetPointIndex(vec_outline, findCenter_points[j][1][0], findCenter_points[j][1][1])
				# grad = all_Concavities[i].ComputeCurvatureGradient(findCenter_points[j][0], pixelStep)
				grad = vec_outline.ComputeCurvatureGradient(myIndex, pixelStep, 1)
				newPoint = array([findCenter_points[j][1][0] + 50*grad[0], findCenter_points[j][1][1] + 50*grad[1]]).astype(int)
				gradients.append([[findCenter_points[j][1][0],newPoint[0]], [findCenter_points[j][1][1],newPoint[1]]])
				newCorner = vut.Vector()
				newCorner.AddDimension('parent')
				newCorner.AddDimension('x')
				newCorner.AddDimension('y')
				newCorner.AddDimension('grad_x')
				newCorner.AddDimension('grad_y')
				newCorner.AddDimension('certainty') #How many possible corner points do we have in this concavity? can be only one
				parentIndex = int(all_Concavities[i].GetName()[-3:])
				newCorner.AddPoint((all_Concavities[i].GetName(), findCenter_points[j][1][0], findCenter_points[j][1][1], grad[0], grad[1], 1.0/len(findCenter_points)))
				newCorner.SetName('con_' + pad_string(parentIndex) + '_cor_' + pad_string(j + 1))
				vec_corners.append(newCorner)
				#corners[findCenter_points[j][1][1], findCenter_points[j][1][0]] = 0.7 #Draw corners directly on image
		print 'Number of corners found: ', len(vec_corners)

		#for i in range(len(vec_corners)):
		#	print vec_corners[i].GetCompletePoint(0)

		# Draw rays-------------------------------------------------------------------
		print '***Drawing lines***'
		# intersections = [] # Vector to keep full scale matrices - one for each corner/ray
		vec_intersections = [] #Vector for all intersections
		for i in range(len(vec_corners)):
			x0 = vec_corners[i].GetPoints_1Dim('x')[0]
			y0 = vec_corners[i].GetPoints_1Dim('y')[0]
			#print 'initial point: [', x0, ',', y0, ']'
			# certainty0 = vec_corners[i].GetPoints_1Dim('certainty')[0]
			grad_x0 = vec_corners[i].GetPoints_1Dim('grad_x')
			#print 'grad_x0', grad_x0
			grad_y0 = vec_corners[i].GetPoints_1Dim('grad_y')
			#print 'grad_y0', grad_y0
			# Compute line eq
			if grad_x0[0] <> 0: 
				k0 = (grad_y0[0]/grad_x0[0])
				m0 = y0 - x0*k0
			else: 
				k0 = 'INF'
				m0 = x0
			for j in range(i + 1, len(vec_corners)):
				x1 = vec_corners[j].GetPoints_1Dim('x')[0]
				y1 = vec_corners[j].GetPoints_1Dim('y')[0]
				#print 'second point: [', x1, ',', y1, ']'
				# certainty1 = vec_corners[j].GetPoints_1Dim('certainty')[0]
				grad_x1 = vec_corners[j].GetPoints_1Dim('grad_x')
				#print 'grad_x1', grad_x1
				grad_y1 = vec_corners[j].GetPoints_1Dim('grad_y')
				#print 'grad_y1', grad_y1
			#
				if grad_x1[0] <> 0: 
					k1 = (grad_y1[0]/grad_x1[0])
					m1 = y1 - x1*k1
				else: 
					k1 = 'INF'
					m1 = x1
				intersect = vectorComputations.ComputeIntersection([k0,m0], [k1,m1])
				intersect_name = vec_corners[i].GetName() + ' with ' + vec_corners[j].GetName()
				# if vec_outline.IsInside('x', intersect[0], 'y', intersect[1]):
				if intersect <> 'NULL':
					newIntersect = vut.Vector()
					con1 = vec_corners[i].GetName()
					con2 = vec_corners[i].GetName()
					if con1 == con2:
						print 'ERROR: Crossing with self'
					newIntersect.AddDimension('parent1') #corner
					newIntersect.AddDimension('parent2') #corner
					newIntersect.AddDimension('distance') # distance between corners
					newIntersect.AddDimension('intersect_x')
					newIntersect.AddDimension('intersect_y')
					newIntersect.AddDimension('certainty') #currently not in use
					cor_1 = [vec_corners[i].GetPoints_1Dim('x')[0], vec_corners[i].GetPoints_1Dim('y')[0]]
					cor_2 = [vec_corners[j].GetPoints_1Dim('x')[0], vec_corners[j].GetPoints_1Dim('y')[0]]
					distance = vectorComputations.ComputeDistance(cor_1, cor_2)
					print 'Distance: ', distance
					newIntersect.AddPoint((vec_corners[i].GetName(), vec_corners[j].GetName(), distance, intersect[0], intersect[1], 1.0))
					newIntersect.SetName(intersect_name)
					vec_intersections.append(newIntersect)
					# intersect.append(intersect_name)
					# intersections.append(intersect)


		print 'Number of intersections found: ', len(vec_intersections)
		for i in range(len(vec_intersections)):
			print vec_intersections[i].GetName()
		# Filter intersections


		# filteredIntersections = []
		vec_filteredIntersection = []
		for i in range(len(vec_intersections)):
			
			# if intersections[i][0] == 260.0:
			if vec_outline.IsInside('x', vec_intersections[i].GetPoints_1Dim('intersect_x')[0], 'y', vec_intersections[i].GetPoints_1Dim('intersect_y')[0]):
				# filteredIntersections.append(intersections[i])
				vec_filteredIntersection.append(vec_intersections[i])
		# intersections = filteredIntersections
		vec_intersections = vec_filteredIntersection

		print 'Number of intersections after filtering: ', len(vec_intersections)
		for i in range(len(vec_intersections)):
			print vec_intersections[i].GetName()
			#corners[intersections[i][1], intersections[i][0]] = 1 #Print intersections directly on image

		# Pair intersections ----------------------------------------------------------
		print '***Filtering intersections***'

		#THIS IS AS FAR AS WE HAVE DONE!!!
		# print 'All concavitiers: ', all_Concavities
		for i in range(len(vec_intersections)):
			print vec_intersections[i].GetName()
			print 'Parent 1 ', vec_intersections[i].GetPoints_1Dim('parent1')
			print 'Parent 2 ', vec_intersections[i].GetPoints_1Dim('parent2')
			print 'X ', vec_intersections[i].GetPoints_1Dim('intersect_x')
			print 'Y ', vec_intersections[i].GetPoints_1Dim('intersect_y')
			print 'Distance ', vec_intersections[i].GetPoints_1Dim('distance')
			
		for i in range(len(all_Concavities)):
			print all_Concavities[i].GetName()

		allowedConcavities = all_Concavities
			
		# Compute confidence matrix

		# Display figures --------------------------------------------------------------
		print '***Printing result***'

		# Zoom
		if crop_:
			if image_:
				mem2 = mem2[300:400,200:300]
				mem = mem[300:400,200:300]
				# corners = corners[300:400,200:300]
			elif shapes_:
				# mem2 = mem2[150:250,150:250]
				mem2 = mem2[250:450,200:400]

		#Add handling for first 5 & last 5
		# print mem.max()
		# print mem.min()
		black = zeros_like(a)

		if plot_all_figures_:
			fig1 = plt.figure()
			plot1 = fig1.add_subplot(111)
			plot1.set_title('Original')
			imshow(a,  cmap='hot', origin='lower')

			
			fig2 = plt.figure()
			plot2 = fig2.add_subplot(111)
			plot2.set_title('Edge')
			imshow(edges,  cmap='gray', origin='lower')

			fig4 = plt.figure()
			plot4 = fig4.add_subplot(111)
			plot4.set_title('Concavity cross product -1')
			imshow(mem,  cmap='hot', origin='lower')

			fig5 = plt.figure()
			plot5 = fig5.add_subplot(111)
			plot5.set_title('After filtering (may be zoomed)')
			myplot = imshow(mem2,  cmap='hot', origin='lower')
			fig5.colorbar(myplot, shrink=0.5, aspect=5)

			# trace2 = traced_edge[300:400,200:300]

		fig3 = plt.figure()
		plot3 = fig3.add_subplot(111)
		for i in range(len(gradients)):
			print gradients[i]
			plt.plot(gradients[i][0], gradients[i][1], linewidth=2.0, color='white')

		for i in range(vec_outline.GetNbrOfPoints()):
			plt.scatter(vec_outline.GetPoints_1Dim('x')[i], vec_outline.GetPoints_1Dim('y')[i], color = 'yellow')


		for i in range(len(vec_corners)):
			#print 'x: ', (vec_corners[i].GetPoints_1Dim('x')[0])
			#print 'y: ', (vec_corners[i].GetPoints_1Dim('y')[0])
			plt.scatter(vec_corners[i].GetPoints_1Dim('x')[0], vec_corners[i].GetPoints_1Dim('y')[0], color='red')
		# for i in range(len(vec_intersections)):
			#print intersections[i]
			#corners[intersections[i][1], intersections[i][0]] = 1 #Print intersections directly on image
			# plt.scatter(vec_intersections[i].GetPoints_1Dim('intersect_x')[0], vec_intersections[i].GetPoints_1Dim('intersect_y')[0], color='white')
		plot3.set_title('Trace with marked corners')
		imshow(black,  cmap='hot', origin='lower')



		# fig6 = plt.figure()
		# plot6 = fig6.add_subplot(111)
		# plot6.set_title('Raypoint1')
		# imshow(all_Rays[1],  cmap='hot', origin='lower')


		# imshow(a, cmap = cm.gray)# left plot in the image above
		plt.show()
