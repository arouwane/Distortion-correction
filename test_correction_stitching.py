#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:21:02 2022

@author: rouwane
"""
import corrector 
import numpy as np 
import scipy as sp 
import os 
import PIL 
import matplotlib.pyplot as plt 
import cv2 
 
def DistortionAdjustment(input_param, cam, images ): 
    """ Getting the registration parameters """ 
    A                    = input_param['A']
    nex                  = input_param['nex']
    ney                  = input_param['ney'] 
    extension            = input_param['extension'] 
    ox                   = input_param['ox']
    oy                   = input_param['oy']
    sigma_gaussian       = input_param['sigma_gaussian']   
    subsampling          = input_param['subsampling']
    interpolation        = input_param['interpolation']
    sx                   = input_param['sx']    
    sy                   = input_param['sy']
    im_type              = input_param['im_type']
    Niter                = input_param['Niter']
    tol                  = input_param['tol']
    modes                = input_param['modes']
    mx                   = input_param['mx']
    my                   = input_param['my']
    d0                   = input_param['d0']
    dire                 = input_param['dire']
    file_name            = input_param['file_name']
 
 
    """ Reading the images """ 
 
    iy = int ( np.ceil(A/ny) )   
    ix = int ( iy%2*(A-(iy-1)*ny) + (iy-1)%2*(ny+1+(iy-1)*ny-A) ) 
    
    a_list  = np.zeros(nex*ney,dtype='int')
    iy_list = np.zeros(nex*ney,dtype='int') 
    ix_list = np.zeros(nex*ney,dtype='int')
    
    k  = 1 
    kk = 0
    if iy%2 ==0:
        for i in range(iy,iy+nex):
            if i%2 == 0  :
                for j in range(ix,ix+ney):
                    xn = j
                    yn = i
                    a = int( yn%2*(xn+(yn-1)*ny) +  (yn-1)%2*( (ny-xn+1) + (yn-1)*ny ) ) 
                    a_list[kk]  = a
                    iy_list[kk] = i 
                    ix_list[kk] = j
                    k+=1 
                    kk+=1
            else: 
                for j in range(ix+ney-1,ix-1,-1):
                    xn = j
                    yn = i
                    a = int( yn%2*(xn+(yn-1)*ny) +  (yn-1)%2*( (ny-xn+1) + (yn-1)*ny ) ) 
                    a_list[kk]  = a
                    iy_list[kk] = i
                    ix_list[kk] = j
                    k+=1 
                    kk+=1
    else :           
        for i in range(iy,iy+nex):
            if i%2 == 0  : 
                for j in range(ix+ney-1,ix-1,-1):
                    xn = j
                    yn = i
                    a = int( yn%2*(xn+(yn-1)*ny) +  (yn-1)%2*( (ny-xn+1) + (yn-1)*ny ) ) 
                    a_list[kk]  = a
                    iy_list[kk] = i
                    ix_list[kk] = j
                    k+=1 
                    kk+=1
            else: 
                for j in range(ix,ix+ney):
                    xn = j
                    yn = i
                    a = int( yn%2*(xn+(yn-1)*ny) +  (yn-1)%2*( (ny-xn+1) + (yn-1)*ny ) ) 
                    a_list[kk]  = a
                    iy_list[kk] = i
                    ix_list[kk] = j
                    k+=1 
                    kk+=1 
                    
    """ Setting the initial translations   """
    first_time = False 
    if images is None: 
        first_time = True 
        images = [None]*nex*ney 
        
    
    for i in range(nex*ney):
        if first_time :
            a = a_list[i]  
            images[i] = corrector.Image(dire+file_name+"%03d" % (a)+'_'+im_type + extension ) 
            images[i].Load() 
            ix = ix_list[i]
            iy = iy_list[i]
            tx = (ix-1) * np.floor( sx - sx*ox/100)
            ty = (iy-1) * np.floor( sy - sy*oy/100)
            images[i].SetCoordinates(ty,tx)
            images[i].SetIndices(iy-1,ix-1)   
        images[i].Load() 
        images[i].GaussianFilter(sigma=sigma_gaussian)
        images[i].BuildInterp(method=interpolation)
        # print('Image'+str(i)+','+str(iy-1)+','+str(ix-1)+','+' ty,tx='+str(images[i].ty)+','+str(images[i].tx))    
            
    tx0 = images[0].tx 
    ty0 = images[0].ty 
    for im in images: 
        im.SetCoordinates(im.tx-tx0,im.ty-ty0)        
        # print(' ty,tx='+str(im.ty)+','+str(im.tx))    
 

 
    regions = [None]* ( nex*(ney-1) + (nex-1)*ney )
    """ Setting the initial overlaps """ 
    k = 0 
    for ix in range(nex):
        for iy in range(ney-1):
            # Setting the horizontal overlaps between image (ix,jx) and (ix,jx+1)
            
            # Left  (reference) image 
            xn = iy + 1
            yn = ix + 1
            a0  = int( yn%2*(xn+(yn-1)*ney) +  (yn-1)%2*( (ney-xn+1) + (yn-1)*ney ) ) 
            
            # Right (deformed) image 
            xn = iy + 2 
            yn = ix + 1 
            a1  = int( yn%2*(xn+(yn-1)*ney) +  (yn-1)%2*( (ney-xn+1) + (yn-1)*ney ) )
            
            regions[k] = corrector.Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'v')
            k +=1  
    # Horizontal overlaps
    for iy in range(ney):
        for ix in range(nex-1):
            # Setting the vertical overlaps between image (ix,jx) and (ix+1,jx)
    
            # Top (reference) image 
            xn = iy + 1
            yn = ix + 1
            a0  = int( yn%2*(xn+(yn-1)*ney) +  (yn-1)%2*( (ney-xn+1) + (yn-1)*ney ) ) 
            
            # Bottom (deformed) image         
            xn = iy + 1
            yn = ix + 2
            a1  = int( yn%2*(xn+(yn-1)*ney) +  (yn-1)%2*( (ney-xn+1) + (yn-1)*ney ) )
            
            regions[k] = corrector.Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'h')
            k+=1

 
    im_list = images 
    r_list  = regions
    
    # We plot the evolution of the correlation score 
    # during the iterations and this for all the overlapping regions 
    Res = np.ones((len(r_list),Niter))*np.nan 
    
    # Vector containing the translations of the images
    # [tx0,ty0,...,txn,tyn]  
    # Connectivity between images and the translation vector 
    # constant, x, y, xy, x², y², x²y, xy², x³, y³ 
    
     
    
    if first_time : 
        
        cam  = corrector.PolynomialCamera(d0, sx/2, sy/2, mx, my)
        
        if modes == 't':
            p0 = d0
            for im in im_list:
                p0 = np.r_[p0,im.tx,im.ty] 
                
        elif modes == 't+d':
            p0 = d0
            for im in im_list:
                p0 = np.r_[p0,im.tx,im.ty] 
                
        elif modes == 'd':
            p0 = d0 
            
        else:
            raise ValueError('Error')        
        p = p0
 
    else:
        p = cam.p 
        for im in im_list:
            p = np.r_[p,im.tx,im.ty] 
        
    nd   = len(cam.p) 
    conn = np.arange(2*len(im_list)).reshape((-1,2)) + nd 
    
    grid = corrector.Grid((nx,ny),(ox,oy),(sx,sy),im_list,r_list,conn)
    
    # grid.ReadTile(dire+'tile_corrected_sigma=0.8.txt')
    
        
    if modes == 't':
        rep  = grid.conn[:,:].ravel()
    elif modes == 't+d': 
        rep  =  np.r_[ np.arange(nd), grid.conn[:,:].ravel() ]
    elif modes == 'd':  
        rep  =  np.arange(nd)
    
    # grid.ReadTile(dire+'TileConfiguration.registered.txt')
    # grid.ReadTile(dire+'TileCorrector.txt')
    
    
    for ik in range(Niter):
        
        # Updating the positions of the overlapping regions 
        for r in grid.regions:
            r.SetBounds(epsilon=2)
            r.IntegrationPts(s=subsampling)  
            
        H,b,res_tot = grid.GetOps(cam) 
     
            
        repk = np.ix_(rep,rep) 
        Hk = H[repk]
        bk = b[rep]
     
        dp = np.linalg.solve(Hk, bk)
        p[rep]  += dp 
     
    
        """ Updating""" 
        # Updating the translation of the im_list 
        for i,im in enumerate(im_list):
            im.SetCoordinates(p[grid.conn[i,0]], p[grid.conn[i,1]]) 
                  
        # Updating the distortion parameters 
        if modes == 't+d' or modes == 'd':
            cam.p = p[:nd] 
        
    
        err = np.linalg.norm(dp)/np.linalg.norm(p) 
        print("Iter # %2d | dp/p=%1.2e" % (ik + 1, err)) 
        for i,r in enumerate(r_list):
            # print("Region # %2d |s=%2.2f" % (i+1,np.std(res_tot[i])) )
            Res[i,ik] = np.std(res_tot[i]) 
        print('Maximal residual std on regions: s=%2.2f '% np.max(Res[:,ik]))
     
        if err < tol:
            break
        
    return cam, images, grid  



#%% 
""" Ep 5-3 11x11 """ 
nx = 10  # Number of images in each direction 
ny = 25
Step = 5

if Step == 0: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step0_220420_154938/intensity/'
    file_name =  'x50-Step0_A'
elif Step == 1: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step1_220421_151742/intensity/'
    file_name =  'x50-Step1_A'
elif Step == 2:
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step2_220422_104635/intensity/'
    file_name =  'x50-Step2_A'
elif Step == 3: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step3_220422_135741/intensity/'
    file_name = 'x50-Step3_A'
elif Step == 4: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step4_220422_180736/intensity/'
    file_name =  'x50-Step4_A'
elif Step == 5: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-Step5_220425_190140/intensity/'
    file_name =  'x50-Step5_A'

else:
    raise ValueError('error')
    
   
    
    

#%% Parameters

# modes = 't with d0'
# modes = 't without d0'
# modes = 'd'
# modes   = 't+d'

d0 = np.array([ 4.32994557e-07, -6.71532622e-07,  5.13778971e-11,  1.46210186e-08,
        6.67696569e-07, -4.94724250e-08, -1.82326908e-08,  8.80300213e-11]) 

d0 = np.zeros(8)
 
input_param = {
    
    "A"              :  32, 
    "nex"            :  4, 
    "ney"            :  4, 
    "ox"             :  20, 
    "oy"             :  20,   
    "interpolation"  :  None, 
    "sigma_gaussian" :  None, 
    "subsampling"    :  None, 
    "Niter"          :  None,  
    "modes"          :  't+d', 
    "tol"            :  1.e-6,
    "mx"             :  [3,5,6,7] , 
    "my"             :  [3,4,6,7] ,
    "d0"             :  d0, # (8 distortion parameters )
    "dire"           :  dire, 
    "file_name"      :  file_name, 
    "sx"             :  1024, 
    "sy"             :  1024,
    "im_type"        :  'I',
    "extension"      :  '.tif' 
    
    }

interpolation_scales = ['linear', 'linear', 'linear', 'cubic-spline']
subsampling_scales   = [3, 2, 1, 1]
sigma_gauss_scales   = [2, 1.2, 0.8, 0.8] 
Niter_scales         = [10, 10, 5, 10]


cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']   = Niter_scales[i]
    cam, images, grid = DistortionAdjustment(input_param, cam0, images0  ) 
    cam0    =  cam 
    images0 =  images 
    
raise ValueError('Stop')
    
#%% 
""" Last stitching step, Getting all the translations  """

d0 = np.array([ 4.28958495e-07, -6.01077789e-07,  1.00164217e-10,  1.47355848e-08,
                6.32162588e-07, -6.22241822e-08, -1.84306000e-08,  1.12761181e-10])
 
input_param = {
    
    "A"              :  1, 
    "nex"            :  10, 
    "ney"            :  25, 
    "ox"             :  20, 
    "oy"             :  20,   
    "interpolation"  :  None, 
    "sigma_gaussian" :  None, 
    "subsampling"    :  None, 
    "Niter"          :  None,  
    "modes"          :  't', 
    "tol"            :  1.e-6,
    "mx"             :  [3,5,6,7] , 
    "my"             :  [3,4,6,7] ,
    "d0"             :  d0, # (8 distortion parameters )
    "dire"           :  dire, 
    "file_name"      :  file_name, 
    "sx"             :  1024, 
    "sy"             :  1024,
    "im_type"        :  'I',
    "extension"      :  '.tif' 
    
    }

interpolation_scales = ['linear', 'linear', 'linear', 'cubic-spline']
subsampling_scales   = [4, 3, 2, 1 ]
sigma_gauss_scales   = [2, 1.2, 0.8, 0.8] 
Niter_scales         = [10, 10, 5, 1]


cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']   = Niter_scales[i]
    cam, images, grid = DistortionAdjustment(input_param, cam0, images0  ) 
    cam0    =  cam 
    images0 =  images 
    
# Set tile relative to the image (0,0)    
tx0 = images[0].tx 
ty0 = images[0].ty 
for im in images: 
    im.SetCoordinates(im.tx-tx0,im.ty-ty0)  
 
 
fusion_mode = 'linear blending'
stitched_output_file = 'Fused_linear_blending'
ims = grid.StitchImages(cam,origin=(0,0), eps=(0,0), fusion_mode=fusion_mode) 

file_fiji = '/media/rouwane/Crucial X6/_Pour Ali/DIC_HR_Ti6242_Tamb/x50-All-Steps-Stitched-Corrected-Fiji/x50-Step0.tif'
ims_fiji = cv2.cvtColor(cv2.imread( file_fiji ), cv2.CV_64F)

imsLap = sp.ndimage.laplace(ims)   

ims_fijiLap = sp.ndimage.laplace(ims_fiji) 

PILimg = PIL.Image.fromarray(np.round(ims).astype("uint8"))
PILimg.save(dire+fusion_mode+'.tif')   

origin = (0,0)
eps    = (0,0)

plt.figure()
# plt.gca().invert_yaxis()
plt.imshow(ims, cmap='gray')
grid.PlotImageBoxes(origin=origin,eps=eps)
plt.axis('equal')
plt.title('Fusion with '+fusion_mode, fontsize=20)
cbar = plt.colorbar()
plt.clim(0,255)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20) 


plt.figure()
# plt.gca().invert_yaxis()
plt.imshow(ims_fiji, cmap='gray')
grid.PlotImageBoxes(origin=origin,eps=eps)
plt.axis('equal')
plt.title('Fusion with '+fusion_mode, fontsize=20)
cbar = plt.colorbar()
plt.clim(0,255)


plt.figure()
plt.imshow(imsLap, cmap='RdBu')
plt.colorbar() 
plt.clim(-1,1)

plt.figure()
plt.imshow(ims_fijiLap, cmap='RdBu')
plt.colorbar() 
plt.clim(-1,1)



#%% 
""" Testing FFT between two images """ 

import skimage
import imreg_dft as ird

skimage.registration.phase_cross_correlation(grid.regions[0].im0.pix[:, -100:],  grid.regions[0].im1.pix[:,:100 ]  )

plt.figure() 
plt.imshow( grid.regions[0].im0.pix[:, -100:], cmap='gray')
plt.colorbar() 

plt.figure() 
plt.imshow( grid.regions[0].im1.pix[:,:100 ], cmap='gray')
plt.colorbar() 



translation(grid.regions[0].im0.pix[:, -100:],  grid.regions[0].im1.pix[:,:100 ]  )

 
result = ird.translation(grid.regions[0].im0.pix ,  grid.regions[0].im1.pix   )
tvec = result["tvec"].round(4)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)



#%% 
""" Visualize the distortion field """ 
x = np.arange(input_param['sx'])
y = np.arange(input_param['sy'])
X,Y = np.meshgrid(x,y,indexing='ij')
xtot = X.ravel() 
ytot = Y.ravel() 

Pxtot,Pytot = cam.P(xtot, ytot)
PxInvTot, PyInvTot = cam.Pinv(xtot, ytot)

plt.figure()
plt.imshow(PxInvTot.reshape(X.shape)-X,cmap='jet')
cbar = plt.colorbar() 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20) 

plt.figure()
plt.imshow(PyInvTot.reshape(Y.shape)-Y,cmap='jet')
cbar = plt.colorbar() 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20)

 
 
#%%  Saving 
""" Save the distortion parameters in a txt file """ 
subdire_pp = ''
param_file = open(dire+subdire_pp+'param_'+str(nex)+'x'+str(ney)+'_A_'+str(A)+'.txt', "w")
param_file.write( 'mx: '+str(mx) + '\n' + 'my: '+str(my) + '\n' + 'p: '+str(cam.p) )
param_file.close()

""" Save the correction fields as a numpy matrix """ 
sx,sy=1024,1024
xc = sx/2; yc = sy/2 

# mx = [3, 5, 6, 7]
# my = [3, 4, 6, 7]
# p = np.array( [ 3.98798546e-07, -1.59927384e-08, -1.11415940e-10,  1.17185153e-08,
#   2.45876015e-06, -2.33290476e-08, -2.49303689e-08 , 8.37707700e-10])
# cam = corrector.PolynomialCamera(p, xc, yc, mx, my)

x = np.arange(sx)
y = np.arange(sy)
X,Y = np.meshgrid(x,y,indexing='ij')
xtot = X.ravel() 
ytot = Y.ravel() 

Pxtot,Pytot = cam.P(xtot, ytot)
""" Saving the correction field on the image domain """
np.save(dire+subdire_pp+str(nex)+'x'+str(ney)+'_A_'+str(A)+'_Cx.npy',Pxtot)
np.save(dire+subdire_pp+str(nex)+'x'+str(ney)+'_A_'+str(A)+'_Cy.npy',Pytot)
# """ Saving the residual fields of each overlap """ 
# np.save(dire+subdire_pp+'residual_'+str(nex)+'x'+str(ney)+'_A_'+str(A)+'.npy',allow_pickle=True)

 
#%% 
""" Saving the corrected sub-images """ 
corrStep = 0 
save_dire = os.path.dirname(os.path.dirname(dire)) + '/intensity-corrected-ParamStep'+str(corrStep)+'/'

if not os.path.exists(save_dire):
   os.makedirs(save_dire)
   print("Saving directory "+save_dire+ "   created")

mx = np.array([3, 5, 6, 7]) 
my = np.array([3, 4, 6, 7]) 
sx, sy = 1024, 1024 
xc = sx/2 ; yc = sy/2 
 
# Step 0 parameters  
p  = np.array([ 4.28924897e-07, -6.00557296e-07,  9.78955523e-11,  1.47382481e-08,
                6.32208782e-07, -6.25285051e-08, -1.84309492e-08,  1.13634487e-10])
cam = corrector.PolynomialCamera(p, xc, yc, mx, my)

x = np.arange(sx)
y = np.arange(sy)
X,Y = np.meshgrid(x,y,indexing='ij')
xtot = X.ravel() 
ytot = Y.ravel() 

Pxtot,Pytot = cam.P(xtot, ytot)

for i,fname in enumerate(os.listdir(dire)):
    print(i,fname)
    im_new = corrector.Image(dire+fname)
    im_new.Load() 
    im_new.BuildInterp()
 
    imc = im_new.Interp(Pxtot, Pytot)
    imc = imc.reshape((sx,sy)) 
    PILimg = PIL.Image.fromarray(np.round(imc).astype("uint8"))
    PILimg.save(save_dire+fname)  
    
    
    # im_new.GaussianFilter(sigma=0.8)
    # PILimg = PIL.Image.fromarray(np.round(im_new.pix).astype("uint8"))
    # PILimg.save(dire+'blured_sigma=0.8/'+fname) 
    
#%% 
""" Stitching """ 
fusion_mode = 'linear blending'
stitched_output_file = 'Fused_linear_blending'
ims = grid.StitchImages(cam,origin=(0,0), eps=(0,0), fusion_mode=fusion_mode) 

PILimg = PIL.Image.fromarray(np.round(ims).astype("uint8"))
PILimg.save(dire+fusion_mode+'.tif')   

origin = (0,0)
eps    = (0,0)

plt.figure()
# plt.gca().invert_yaxis()
plt.imshow(ims, cmap='gray')
grid.PlotImageBoxes(origin=origin,eps=eps)
plt.axis('equal')
plt.title('Fusion with '+fusion_mode, fontsize=20)
cbar = plt.colorbar()
plt.clim(0,255)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20) 
 