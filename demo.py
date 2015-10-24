import numpy as np
from pylab import *
import intersection_finder as intFind
# from scipy.optimize import fsolve as sol

# Helper functions--------------------------------------------------------------
			



fileIn = 'C:\\Python\\Test\\myFile.bmp'

img = imread(fileIn)

# Set parameters ------------------------------------------------------------------
print '***Set parameters in***'


dim = 512
print 'Dimension is: ', dim
noise_ = 0
shapes_ = 0
image_ = 1
grid_ = 0
crop_ = 0
print_all_ = 0


pixelStep = 7
print 'Calculation scale is: ', pixelStep

# def DrawGradient(img, point, gradient):


if grid_:
	#Display grid to check colomap
	a = np.zeros((dim, dim)).astype(float) #unsigned integer type needed by watershed
	for i in range(dim):
		a[:,i] = 1.0*i/dim #Normalisera till [0,1]
	fig1 = plt.figure()
	plot1 = fig1.add_subplot(111)
	plot1.set_title('Original')
	imshow(a,  cmap='hot', origin='lower')
	plt.show()

if image_:
	#Use saved image
	a = img[:,:,0].astype(uint8)
	print 'Using image: ', fileIn
else:
	#Generate zero matrix
	a = np.zeros((dim, dim)).astype(uint8) #unsigned integer type needed by watershed

if noise_:
	#Add noise
	b = rnd.randint(200, size=a.shape) #Random values [0,1]
	# a += b
	# a = where(b >= 195 , 1, 0)

if shapes_:
	#Add shapes
	y, x = np.ogrid[0:dim, 0:dim]
	print y.shape
	m1 = ((y-200)**2 + (x-300)**2 < 100**2)
	m2 = ((y-300)**2 + (x-250)**2 < 100**2)
	m2_2 = ((y-250)**2 + (x-370)**2 < 100**2)
	m3 = ((y-400)**2 + (x-300)**2 < 50**2)
	m3_2 = ((y-400)**2 + (x-300)**2 < 100**2)
	m4 = ((y-450)**2 + (x-100)**2 < 50**2)
	m5 = ((y-100)**2 + (x-100)**2 < 20**2) 
	m6 = ((y-120)**2 + (x-100)**2 < 20**2)
	# a[m2_2+m2+m3_2]=1 # Shamrock of three
	# a[m1 + m2 + m3] = 1 # Stack of three, differing size
	# a[m5 + m6] = 1 # Small stack of two
	a[m1 + m2 + m2_2 + m3_2] = 1 #Shamrock of four


intersectionFinder = intFind.IntersectionFinder()
intersectionFinder.Run(a, pixelStep, print_all_)

