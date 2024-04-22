import json 
import sys 
import os 
sys.path.append('./src')
import numpy as np 
import matplotlib.pyplot as plt 
from classProjector import Projector 
import corrector 
import PIL 

"""
Reading the parameters 
of a previous distortion identifications 
saved in a json file 
and correcting the different sub-images 
in a folder 
No stitching is performed in this example
"""

file = 'examples/data/lscm_speckle/x100_3x4_10pct2/identif_case4.json'

data = json.load(open(file))
parameters = data['parameters']
proj = data['projector']
p =  np.array(proj['p'])
mx = proj['mx']
my = proj['my'] 
cam =  Projector(p, parameters['sx']/2, parameters['sy']/2, mx, my, parameters['sx'], parameters['sy'])
cam.ShowValues()

cam.PlotGrid(size=[parameters['sx'], parameters['sy']], nelemx=20, alpha=5)
plt.show() 

#%% 
""" Correcting all the images and saving them in a subdirectory  """
save_dire = parameters['dire']+'corrected_case4/'
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
