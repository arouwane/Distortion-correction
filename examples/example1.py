import sys 
sys.path.append('./src')
import corrector 
import matplotlib.pyplot as plt 
import os 

""" 
Distortion identification from the image mosaic 
Input: Grid of images of size (nx,ny)
Output: A distortion function 
""" 

dire = 'examples/data/artificial_speckle/' 
file_name = 'speckle_image_A' 
 
input_param = {
"nx"             :  3, 
"ny"             :  3,
"A"              :  1, 
"nex"            :  3, 
"ney"            :  3, 
"ox"             :  20, 
"oy"             :  20,    
"interpolation"  :  'cubic-spline', 
"sigma_gaussian" :  0.8, 
"subsampling"    :  None, 
"Niter"          :  None,  
"modes"          :  't+d', 
"tol"            :  1.e-4,
# Modes          : constant, x, y, x*y, x**2, y**2, x**2*y, x*y**2, x**3, y**3
"mx"             :  [3, 4, 5,  7, 8], 
"my"             :  [3, 4, 5,  6, 9],
"d0"             :  None, 
"dire"           :  dire, 
"file_name0"     :  file_name,
"file_name1"     :  '',
"ndigits"        :  2,   
"sx"             :  1024, 
"sy"             :  1024, 
"extension"      :  '.tif' 
}



interpolation_scales = ['bilinear', 'cubic-spline','cubic-spline']
subsampling_scales   = [3,2,1]
sigma_gauss_scales   = [2,1,0.8] 
Niter_scales         = [30,30,20] 
 

cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']          = Niter_scales[i]
    cam, images, grid, res_tot =  corrector.DistortionAdjustment(input_param, cam0, images0, epsilon = 10  ) 
    cam0              =  cam 
    images0           =  images 
 
 #%% 
""" Measured distortion"""
cam.ShowValues() 

""" Visualization of the measured distortion function """
cam.Plot(size=[input_param['sx'],input_param['sy']]) 

cam.PlotGrid(size=(1024,1024), nelemx=10, alpha=5)

cam.PlotInverseGrid(size=(1024,1024), nelemx=10, alpha=5)

plt.show() 