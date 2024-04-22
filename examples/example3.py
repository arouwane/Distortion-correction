import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import skimage 
import scipy as sp 
import scipy.interpolate as spi 
import os  

"""
Parametric study corresponding to 
Table 2 of article  
Computing the RMS of strain and 
gray-level residual of the registration of images 
of the same sample
"""
#%% 

def Register(file0, file1, N=1 ):
    """
    file0: Reference image file
    file1: Deformed  image file  
    N: subsampling factor of the strain field 
    returns displacement and strain fields 
    """
    f0  =  cv2.cvtColor( cv2.imread(file0), cv2.COLOR_BGR2GRAY)    # ; f0 = np.copy(f0[:3500,:5500])
    f1  =  cv2.cvtColor( cv2.imread(file1), cv2.COLOR_BGR2GRAY) 
  
    sizeim = np.min(np.vstack((f0.shape,f1.shape)), axis=0)
    repk = np.ix_(np.arange(sizeim[0]),np.arange(sizeim[1]))
    im0 = f0[repk]
    im1 = f1[repk]
    # Performing registration using disoptical flow 
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    u = flow.calc(im0,im1, None)

    ux  = u[::,::,1]
    uy  = u[::,::,0]

    uxss  =  ux[::N,::N]  
    uyss  =  uy[::N,::N]

    exx = np.gradient(uxss,axis=0) / N   
    eyy = np.gradient(uyss,axis=1) / N    
    exy = 0.5 * ( np.gradient(uyss,axis=0) + np.gradient(uxss,axis=1) ) / N


    # Residual analysis 
    x = np.arange(0, im0.shape[0])
    y = np.arange(0, im0.shape[1])
    X,Y = np.meshgrid(x, y, indexing='ij')
 
    # bilinear_interpolator1 = spi.RegularGridInterpolator((x,y), im1) # Bilinear interpolator 
    spline_interpolator1  = spi.RectBivariateSpline(x, y, im1)       # Cubic spline interpolator 

    u1 = X.ravel() + ux.ravel()  
    v1 = Y.ravel() + uy.ravel()
    u1[(u1<0)] = 0 
    u1[u1>im0.shape[0]-1] = im0.shape[0]-1 
    v1[(v1<0)] = 0 
    v1[v1>im0.shape[1]-1] = im0.shape[1]-1 

    # im1oidu = bilinear_interpolator1(np.vstack((u1,v1)).T).reshape(im0.shape)    # Computing   im1(x+u01(x))
    im1oidu = spline_interpolator1.ev(u1,v1).reshape(im0.shape)  

    res   = im0 - im1oidu   
    resN  = im0 - np.mean(im0) - np.std(im0)/np.std(im1oidu)*(im1oidu-np.mean(im1oidu))   # For correcting the difference of brightness 
    return ux, uy, exx, eyy, exy, res, resN   

def CropFields(fields, c):
    croped_fields = [None]*len(fields)
    for i,f in enumerate(fields):
        croped_fields[i] = f[c:-c , c:-c]
    return croped_fields 

def ComputeRMS(fields):
    rms_fields = [None]*len(fields) 
    n = fields[0].shape[0] * fields[0].shape[1] 
    for i, f in enumerate(fields):
        rms_fields[i] = np.sqrt( np.sum(f**2) / n )
    return rms_fields  

def ComputeRMS_Overlaps(nx, ny, sx, sy, fields, c):
    
    xc1 = np.arange(1,nx)*sx/nx ; xc1 = xc1.astype('int')
    yc1 = np.arange(1,ny)*sy/ny ; yc1 = yc1.astype('int')
 
    xc = np.array([])
    yc = np.array([])
    for i in range(len(xc1)):
        xc = np.r_[ xc, np.arange(xc1[i] - c, xc1[i] + c) ]
    for i in range(len(yc1)):
        yc = np.r_[ yc, np.arange(yc1[i] - c, yc1[i] + c) ]     
    xc = xc.astype('int')
    yc = yc.astype('int') 

    rms_fields = [0]*len(fields) 
    for j,f in enumerate(fields):
        for i in range(len(xc)): 
            rms_fields[j] += np.sum( f[ xc[i], : ]**2 )
    for j,f in enumerate(fields):
        for i in range(len(yc)):
            rms_fields[j] += np.sum( f[ : , yc[i] ]**2 ) 

    # n = fields[0].shape[1] * len(xc) + fields[0].shape[0] * len(yc)
    n = fields[0].shape[1] * len(xc) + fields[0].shape[0] * len(yc)
    for j in range(len(rms_fields)):
        rms_fields[j] = np.sqrt( rms_fields[j] / n )  
    
    return rms_fields 



emin = -0.006 
emax =  0.006 
 

case = 'case4'

if case == 'uncorrected old':
    file0 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct1/intensity/Fused_uncorrected_linear_blending.tif' 
    file1 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct2/intensity/Fused_uncorrected_linear_blending.tif'
elif case == 'case1 old':
    file0 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct1/intensity/Fused_corrected_linear_blending_1.tif' 
    file1 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct2/intensity/Fused_corrected_linear_blending_1.tif'
elif case == 'case2 old':
    file0 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct1/intensity/Fused_corrected_linear_blending_2.tif' 
    file1 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct2/intensity/Fused_corrected_linear_blending_2.tif'
elif case == 'case3 old': 
    file0 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct1/intensity/Fused_corrected_linear_blending_3.tif' 
    file1 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct2/intensity/Fused_corrected_linear_blending_3.tif'
elif case == 'case4 old':
    file0 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct1/intensity/Fused_corrected_linear_blending_4.tif' 
    file1 = 'C:/Users/Ali/Downloads/x100_4x3/x100_4x3_10pct2/intensity/Fused_corrected_linear_blending_4.tif'

elif case == 'uncorrected': 
    file0 = 'examples/data/lscm_speckle/x100_3x4_10pct1/mosaic_uncorrected_fiji.tif'
    file1 = 'examples/data/lscm_speckle/x100_3x4_10pct2/mosaic_uncorrected_fiji.tif' 
elif case == 'case1' :
    file0 = 'examples/data/lscm_speckle/x100_3x4_10pct1/corrected_case1/Fused.tif'
    file1 = 'examples/data/lscm_speckle/x100_3x4_10pct2/corrected_case1/Fused.tif'  
elif case == 'case2' : 
    file0 = 'examples/data/lscm_speckle/x100_3x4_10pct1/corrected_case2/Fused.tif'
    file1 = 'examples/data/lscm_speckle/x100_3x4_10pct2/corrected_case2/Fused.tif' 
elif case == 'case3' :
    file0 = 'examples/data/lscm_speckle/x100_3x4_10pct1/corrected_case3/Fused.tif'
    file1 = 'examples/data/lscm_speckle/x100_3x4_10pct2/corrected_case3/Fused.tif' 
elif case == 'case4' : 
    file0 = 'examples/data/lscm_speckle/x100_3x4_10pct1/corrected_case4/Fused.tif'
    file1 = 'examples/data/lscm_speckle/x100_3x4_10pct2/corrected_case4/Fused.tif' 
else: 
    raise ValueError('Error in case')
 

ux, uy, exx, eyy, exy, res, resN = Register(file0, file1) 
uxc, uyc, exxc, eyyc, exyc, resc, resNc = CropFields([ux, uy, exx, eyy, exy, res, resN], 200)

 
sx = uxc.shape[0]
sy = uxc.shape[1]
rms_eyy_o, rms_exx_o, rms_exy_o, rms_resN_o = ComputeRMS_Overlaps(nx=3, ny=4, sx=sx, sy=sy, 
                                                          fields=[eyyc, exxc, exyc, resNc], c=30)


print(case)
print('RMS on overlapping regions')
print(rms_eyy_o, rms_exx_o, rms_exy_o, rms_resN_o ) 


