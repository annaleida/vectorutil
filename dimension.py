class Dimension:

	name_ = 'dimension'
	points_ = []

	def __init__(self, name):
		self.name_ = name
		self.points_ = []

	def GetName(self):
		return self.name_
	def GetPoints(self):
		return self.points_

