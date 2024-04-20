import sys 
sys.path.append('./src')
import corrector 
import matplotlib.pyplot as plt 
import os 
import json 
import numpy as np 
import PIL 

""" 
Example: 
A set of 3x3 mosaic distorted images of a speckle pattern 
is given in folder examples/data/artificial_speckle
This scripts show how to identify the distortion function  
""" 

test_case = 'case4'
results_file = 'identif_' + test_case + '.json'
parameters = {
"nx"                   :  3, 
"ny"                   :  4,
"A"                    :  1, 
"nex"                  :  3, 
"ney"                  :  4, 
"ox"                   :  10, 
"oy"                   :  10,    
"interpolation_scales" :  ['bilinear', 'cubic-spline', 'cubic-spline'],   
"subsampling_scales"   :  [3,2,1], 
"sigma_gauss_scales"   :  [2,1,0.8],  
"Niter_scales"         :  [30,30,10], 
"modes"                :  't+d', 
"tol"                  :  1.e-5,
"mx"                   :  [3, 4, 5, 6, 7, 9], 
"my"                   :  [3, 4, 5, 6, 7, 8],
"d0"                   :  None, 
"dire"                 :  'examples/data/lscm_speckle/x100_3x4_10pct2/',   
"file_name0"           :  'x100_epr-jerem_10pct_002_A', 
"file_name1"           :  '_I', 
"ndigits"              :  3,   
"sx"                   :  1024, 
"sy"                   :  1024, 
"extension"            :  '.tif' 
}

""" Running the identification procedure""" 
cam, images, grid, res_tot = corrector.DistortionAdjustement_Multiscale(parameters,
                                                                         cam0=None, images0=None, epsilon=10)
""" Measured distortion"""
cam.ShowValues() 

""" Visualization of the measured distortion function """
cam.Plot(size=[parameters['sx'], parameters['sy']]) 

cam.PlotGrid(size=(1024,1024), nelemx=10, alpha=5)

cam.PlotInverseGrid(size=(1024,1024), nelemx=10, alpha=5)

plt.show() 


#%% 
""" Saving the input parameters and results """
results = {
    "parameters" : parameters, 
    "projector":  {'mx': cam.mx,
                   'my': cam.my, 
                   'p' : list(cam.p) }  
}
with open(parameters['dire']+results_file, "w") as json_file:
    json.dump(results, json_file)

 
#%% 
""" Stitching the image mosaic with the built-in functions 
One can also stitch the images after distortion correction using
other stitching tools such as the fiji plugin 
"""
fname = 'mosaic_'

grid.ExportTile(parameters['dire']+'TileConfiguration_'+test_case + '.txt')  # Exporting Tile 

ims_unc = grid.StitchImages(fusion_mode='average') # Stitching the mosaic without correction 
PILimg = PIL.Image.fromarray(np.round(ims_unc).astype("uint8"))
PILimg.save(parameters['dire']+fname+ test_case + parameters['extension'])  
 