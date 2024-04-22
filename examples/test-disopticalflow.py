import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
from skimage.filters import gaussian

file0 = 'examples/data/lsclm_tensile_test/Ref/mosaic.tif' 
file1 = 'examples/data/lsclm_tensile_test/State2/mosaic.tif' 
 
f0  =  cv2.cvtColor( cv2.imread(file0), cv2.COLOR_BGR2GRAY) 
f1  =  cv2.cvtColor( cv2.imread(file1), cv2.COLOR_BGR2GRAY) 

sizeim = np.min(np.vstack((f0.shape,f1.shape)), axis=0)
repk = np.ix_(np.arange(sizeim[0]),np.arange(sizeim[1]))
im0 = f0[repk]
im1 = f1[repk]

flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
# flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
# flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
# MANUAL
# flow = cv2.DISOpticalFlow_create()
# flow.setVariationalRefinementAlpha(20.0)		# Laplacian of displacment
# flow.setVariationalRefinementGamma(10.0)		# Gradient of image consistency
# flow.setVariationalRefinementDelta(5.0) 	    # Optical flow
# flow.setVariationalRefinementIterations(5)	    # Number of iterations
# flow.setFinestScale(0)
# flow.setPatchSize(13)
# flow.setPatchStride(7)
print('DISOpticalFlow parameters')
print(flow.getVariationalRefinementAlpha()	)	# Laplacian of displacment
print(flow.getVariationalRefinementGamma()	)	# Gradient of image consistency
print(flow.getVariationalRefinementDelta() 	)    # Optical flow
print(flow.getVariationalRefinementIterations()	 )   # Number of iterations
print(flow.getFinestScale())
print(flow.getPatchSize())
print(flow.getPatchStride())

u = flow.calc(im0,im1, None)

ux  = u[::,::,1]
uy  = u[::,::,0]

exx = np.gradient(ux,axis=0)    
eyy = np.gradient(uy,axis=1)    
exy = 0.5 * ( np.gradient(uy,axis=0) + np.gradient(ux,axis=1) )

exx = exx[200:-200, 200:-200]
eyy = eyy[200:-200, 200:-200]
exy = exy[200:-200, 200:-200]

emin = -0.02
emax =  0.02 

plt.figure()
plt.imshow(exx, cmap='RdBu' )
plt.clim(emin, emax)
plt.colorbar() 

plt.figure()
plt.imshow(eyy, cmap='RdBu' )
plt.clim(emin, emax)
plt.colorbar()  

plt.figure()
plt.imshow(exy, cmap='RdBu' )
plt.clim(emin, emax)
plt.colorbar() 

plt.show() 

 