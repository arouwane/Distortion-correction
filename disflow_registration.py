#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 08:45:44 2023

@author: rouwane
"""

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os 


Step = 5

file0 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step0_220420_154938/intensity-corrected-ParamStep0/x50-Step0_Fused_I.tif'
 
if Step   == 1: 
    file1 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step1_220421_151742/intensity-corrected-ParamStep1/x50-Step1_Fused_I.tif'
elif Step == 2:
    file1 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step2_220422_104635/intensity-corrected-ParamStep2/x50-Step2_Fused_I.tif'
elif Step == 3: 
    file1 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step3_220422_135741/intensity-corrected-ParamStep3/x50-Step3_Fused_I.tif'
elif Step == 4: 
    file1 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step4_220422_180736/intensity-corrected-ParamStep4/x50-Step4_Fused_I.tif'
elif Step == 5: 
    file1 = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step5_220425_190140/intensity-corrected-ParamStep5/x50-Step5_Fused_I.tif'
else: 
    raise ValueError('error')
    

f0  =  cv2.cvtColor( cv2.imread(file0), cv2.COLOR_BGR2GRAY)    # ; f0 = np.copy(f0[:3500,:5500])
f1  =  cv2.cvtColor( cv2.imread(file1), cv2.COLOR_BGR2GRAY)    

sizeim = np.min(np.vstack((f0.shape,f1.shape)), axis=0)
repk = np.ix_(np.arange(sizeim[0]),np.arange(sizeim[1]))
im0 = f0[repk]
im1 = f1[repk]
 

flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)



flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow.setFinestScale(0)


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

print(flow.getVariationalRefinementAlpha()	)	# Laplacian of displacment
print(flow.getVariationalRefinementGamma()	)	# Gradient of image consistency
print(flow.getVariationalRefinementDelta() 	)    # Optical flow
print(flow.getVariationalRefinementIterations()	 )   # Number of iterations
print(flow.getFinestScale())
print(flow.getPatchSize())
print(flow.getPatchStride())





u = flow.calc(im0,im1, None)

ux  = u[::,::,1]; uy  = u[::,::,0]

plt.figure()
plt.imshow(ux)
 
# c = 100
# ux = ux[c:-c,c:-c];  uy = uy[c:-c,c:-c]

 
d_ux_dx = cv2.Sobel(ux,cv2.CV_64F,1,0,ksize=-1)
d_ux_dy = cv2.Sobel(ux,cv2.CV_64F,0,1,ksize=-1)
d_uy_dx = cv2.Sobel(uy,cv2.CV_64F,1,0,ksize=-1)
d_uy_dy = cv2.Sobel(uy,cv2.CV_64F,0,1,ksize=-1)

exx = d_ux_dx 
eyy = d_uy_dy 
exy = 0.5 * ( d_ux_dy + d_uy_dx )


flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

u = flow.calc(im0,im1, None)

N = 1 

Ux  = u[::,::,0]
Uy  = u[::,::,1]

Uxss  =  Ux[::N,::N]  
Uyss  =  Uy[::N,::N] 

EXX = np.gradient(Uxss,axis=1)/N
EYY = np.gradient(Uyss,axis=0)/N
EXY = 0.5 * (np.gradient(Uxss,axis=0) + np.gradient(Uyss,axis=1)) / N

save_dire = os.path.dirname(file1) + '/disflow-reg/' 
output = 'x50-Step'+str(Step)

if not os.path.exists(save_dire):
   os.makedirs(save_dire)
   print("Saving directory "+save_dire+ "   created")
   

cv2.imwrite(save_dire+"Ux_"+output+"_SS="+str(N)+".tiff",Uxss)
cv2.imwrite(save_dire+"Uy_"+output+"_SS="+str(N)+".tiff",Uyss)
cv2.imwrite(save_dire+"EXX_"+output+"_SS="+str(N)+".tiff",EXX)
cv2.imwrite(save_dire+"EYY_"+output+"_SS="+str(N)+".tiff",EYY)
cv2.imwrite(save_dire+"EXY_"+output+"_SS="+str(N)+".tiff",EXY)

cv2.imwrite(save_dire+"Ux_SCALE0_"+output+"_SS="+str(N)+".tiff",Uxss)
 

