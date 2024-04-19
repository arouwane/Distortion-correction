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

parameters = {
"nx"                   :  3, 
"ny"                   :  3,
"A"                    :  1, 
"nex"                  :  3, 
"ney"                  :  3, 
"ox"                   :  20, 
"oy"                   :  20,    
"interpolation_scales" :  ['bilinear', 'cubic-spline', 'cubic-spline'],   
"subsampling_scales"   :  [3,2,1], 
"sigma_gauss_scales"   :  [2,1,0.8],  
"Niter_scales"         :  [30,30,10], 
"modes"                :  't+d', 
"tol"                  :  1.e-4,
"mx"                   :  [3, 4, 5, 7, 8], 
"my"                   :  [3, 4, 5, 6, 9],
"d0"                   :  None, 
"dire"                 :  'examples/data/artificial_speckle/', 
"file_name0"           :  'speckle_image_A',
"file_name1"           :  '',
"ndigits"              :  2,   
"sx"                   :  1024, 
"sy"                   :  1024, 
"extension"            :  '.tif' 
}

results_file = 'identif.json'


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
""" Correcting all the images and saving them in a subdirectory  """
save_dire = parameters['dire']+'corrected_'
sigma_gaussian = 0 # Parameter for Gaussian bluring of the images 

x = np.arange(parameters['sx'])
y = np.arange(parameters['sy'])
X,Y = np.meshgrid(x,y,indexing='ij')
xtot = X.ravel() 
ytot = Y.ravel() 

Pxtot,Pytot = cam.P(xtot, ytot)


for i,fname in enumerate(os.listdir(parameters['dire'])):
    print(i,fname)
    if fname[-3:] == 'tif': 
        im_new = corrector.Image(parameters['dire']+fname)
        im_new.Load() 
        if im_new.pix.shape == (parameters['sx'],parameters['sy']): 
            im_new.BuildInterp()
            im_new.GaussianFilter(sigma = sigma_gaussian)
            imc = im_new.Interp(Pxtot, Pytot)
            imc = imc.reshape((parameters['sx'],parameters['sy'])) 
            PILimg = PIL.Image.fromarray(np.round(imc).astype("uint8"))
            PILimg.save(save_dire+fname)  
        else: 
            print("Skip file")
    else: 
        print("Skip file")


#%% 
""" Stitching the image mosaic with the built-in functions 
One can also stitch the images after distortion correction using
other stitching tools such as the fiji plugin 
"""
fname = 'mosaic'

grid.ExportTile('TileConfiguration.txt')  # Exporting Tile 

ims_unc = grid.StitchImages(fusion_mode='average') # Stitching the mosaic without correction 
PILimg = PIL.Image.fromarray(np.round(ims_unc).astype("uint8"))
PILimg.save(parameters['dire']+fname+'_uncorrected'+parameters['extension'])  

ims_c = grid.StitchImages(cam= cam, fusion_mode='average') 
PILimg = PIL.Image.fromarray(np.round(ims_c).astype("uint8"))
PILimg.save(parameters['dire']+fname+'_corrected'+parameters['extension'])  
 