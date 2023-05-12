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

def CreateGrid(input_param, images=None):
    A                    = input_param['A']
    nex                  = input_param['nex']
    ney                  = input_param['ney'] 
    extension            = input_param['extension'] 
    ox                   = input_param['ox']
    oy                   = input_param['oy']
    sigma_gaussian       = input_param['sigma_gaussian']   
    interpolation        = input_param['interpolation']
    sx                   = input_param['sx']    
    sy                   = input_param['sy']
    im_type              = input_param['im_type']
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
    
    grid = corrector.Grid((nx,ny),(ox,oy),(sx,sy),images,regions)
    return grid 
            
 
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
        

    
    grid = corrector.Grid((nx,ny),(ox,oy),(sx,sy),im_list,r_list)
    
    nd   = len(cam.p) 
    conn = np.arange(2*len(im_list)).reshape((-1,2)) + nd 
    grid.Connectivity(conn)
    
    # grid.ReadTile(dire+'tile_corrected_sigma=0.8.txt')
    
        
    if modes == 't':
        rep  = grid.conn[:,:].ravel()
    elif modes == 't+d': 
        rep  =  np.r_[ np.arange(nd), grid.conn[:,:].ravel() ]
    elif modes == 'd':  
        rep  =  np.arange(nd)
    
    # grid.ReadTile(dire+'TileConfiguration.registered.txt')
    # grid.ReadTile(dire+'TileCorrector.txt')
    
    print('--GN')
    for ik in range(Niter):
        
        # Updating the positions of the overlapping regions 
        for r in grid.regions:
            r.SetBounds(epsilon=2)
            r.IntegrationPts(s=subsampling)  
            
        H,b,res_tot = grid.GetOps(cam) 
     
            
        repk = np.ix_(rep,rep) 
        Hk = H[repk]
        bk = b[rep]
        
        # Hkinv = np.linalg.inv(Hk)
        # return Hkinv 
        # raise ValueError('Stop')
        
     
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
        print('----------------------------------------')
        print("Iter # %2d | dp/p=%1.2e" % (ik + 1, err)) 
        for i,r in enumerate(r_list):
            # print("Region # %2d |s=%2.2f" % (i+1,np.std(res_tot[i])) )
            Res[i,ik] = np.std(res_tot[i]) 
        print('Maximal residual std on regions: %2.2f '% np.max(Res[:,ik]))
        print('Minimal residual std on regions: %2.2f '% np.min(Res[:,ik]))
        print('Mean residual std on regions: %2.2f' % np.mean(Res[:,ik]) )
        print('Std residual std on regions: %2.2f' % np.std(Res[:,ik]))
        print('----------------------------------------')
        if err < tol:
            break
        
    return cam, images, grid, res_tot  



#%% Ti6242 
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

#%%  Ep 5-3 11x11 

nx = 11  # Number of images in each direction 
ny = 11
Step = 1

if Step == 0:  
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_3D_Confocal_prio/5-3_ref/x100_11x11_stitch_210817_101427/intensity/' 
    file_name = 'x100_11x11_stitch_A'
    # d0 = np.array([ 3.52150029e-07,  3.61382915e-07,  1.05387843e-10,  1.04280200e-08, 1.94910898e-06,  1.91450928e-07, -2.45972757e-08,  5.65922014e-11])    
elif Step == 1: 
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_3D_Confocal_prio/5-3_etap1/5-3_Rp0.1__210820_094537/intensity/' 
    file_name = '5-3_Rp0.1__000_A'
    # d0 = np.array([ 3.53665304e-07, -3.95685470e-08, -8.99430899e-12,  1.02952874e-08, 1.95576850e-06, -8.30114438e-08, -2.46524176e-08,  3.02392438e-12])
elif Step == 2:
    dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_3D_Confocal_prio/5-3_etap2/210820_162553/intensity/'
    file_name = '210820_1625_000_A' 
    # d0 = np.array([ 3.43471798e-07,  5.40200090e-07,  2.72129188e-10,  1.04229904e-08, 3.90696908e-07,  6.75052810e-07, -2.15127081e-08,  1.26789956e-10])

else:
    raise ValueError('error')

  
    

#%% Parameters

# modes = 't with d0'
# modes = 't without d0'
# modes = 'd'
# modes   = 't+d'
 

d0 = np.zeros(8)
 
input_param = {
    
    "A"              :  49, 
    "nex"            :  5, 
    "ney"            :  5, 
    "ox"             :  10, 
    "oy"             :  10,   
    "interpolation"  :  'cubic-spline', 
    "sigma_gaussian" :  0, 
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
Niter_scales         = [20, 20, 10, 5]


cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']          = Niter_scales[i]
    cam, images, grid, res_tot =  DistortionAdjustment(input_param, cam0, images0  ) 
    cam0              =  cam 
    images0           =  images 
    
raise ValueError('Stop')

 


grid = CreateGrid( input_param ) 

cam = corrector.PolynomialCamera(d0, 512, 512, [3,5,6,7], [3,4,6,7])

grid.SetPairShift(cam, overlap=[10,10]) 


#%%  Sensitivity analysis 

d0 = np.zeros(10) 
 
input_param = {
    
    "A"              :  49, 
    "nex"            :  3, 
    "ney"            :  3, 
    "ox"             :  10, 
    "oy"             :  10,   
    "interpolation"  :  'cubic-spline', 
    "sigma_gaussian" :  0, 
    "subsampling"    :  None, 
    "Niter"          :  None,  
    "modes"          :  't+d', 
    "tol"            :  1.e-6,
    "mx"             :  [3,5,6,7,9] , 
    "my"             :  [3,4,6,7,8] ,
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
Niter_scales         = [20, 20, 10, 5]


cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']          = Niter_scales[i]
    cam, images, grid, res_tot =  DistortionAdjustment(input_param, cam0, images0  ) 
    cam0              =  cam 
    images0           =  images 
    
raise ValueError('Stop')


        
nd = len(d0)
rep  =  np.arange(10)
H,b,res_tot = grid.GetOps(cam) 
   
repk = np.ix_(rep,rep) 
Hk = H[repk]
bk = b[rep]
    
Hinv = np.linalg.inv(H)

plt.figure()
plt.imshow(Hinv[10:20,10:20]) 
plt.colorbar()
plt.clim(-1.e-5,1.e-6)
plt.title(r'$1, x, y, xy, x^2, y^2, x^2y, xy^2, x^3, y^3$')



    
#%% 
""" Last stitching step, Getting all the translations  """

d0  = np.zeros(8)

d0 = np.array([ 3.96763078e-07,  2.39404586e-08, -1.01131001e-10,  1.07496071e-08,
        2.54048164e-06, -1.07787169e-07, -2.56639308e-08,  2.16932626e-10])
 
input_param = {
    
    "A"              :  1, 
    "nex"            :  11, 
    "ney"            :  11, 
    "ox"             :  10, 
    "oy"             :  10,   
    "interpolation"  :  None, 
    "sigma_gaussian" :  None, 
    "subsampling"    :  None, 
    "Niter"          :  None,  
    "modes"          :  't', 
    "tol"            :  1.e-8,
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
 
interpolation_scales = ['linear','linear','linear', 'linear','linear']
subsampling_scales   = [ 5, 4, 3, 2, 1 ]
sigma_gauss_scales   = [ 4, 3, 2, 1, 0.8] 
Niter_scales         = [ 5, 5, 5, 5, 20 ]
 
cam0    = None 
images0 = None 

for i in range(len(subsampling_scales)):
    print('*********** SCALE '+str(i+1)+' ***********')
    input_param['interpolation']  = interpolation_scales[i]
    input_param['subsampling']    = subsampling_scales[i]
    input_param['sigma_gaussian'] = sigma_gauss_scales[i]
    input_param['Niter']   = Niter_scales[i]
    cam, images, grid, res_tot = DistortionAdjustment(input_param, cam0, images0  ) 
    cam0    =  cam 
    images0 =  images 
    
plt.figure()
grid.PlotResidualMap(res_tot)
plt.colorbar()
plt.clim(-10,10)    

#%% 
    
    
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

skimage.registration.phase_cross_correlation(grid.regions[0].im0.pix[:, -70:],  grid.regions[0].im1.pix[:,:70 ],  )

skimage.registration.phase_cross_correlation(grid.regions[0].im0.pix,  grid.regions[0].im1.pix, overlap_ratio=0.3  )





plt.figure() 
plt.imshow( image1, cmap='gray')
plt.colorbar() 

plt.figure() 
plt.imshow( image2, cmap='gray')
plt.colorbar() 

image1FFT = np.fft.fft2(image1)
image2FFT = np.conjugate( np.fft.fft2(image2) )

imageCCor = np.real( np.fft.ifft2( (image1FFT*image2FFT) ) )
imageCCorShift = np.fft.fftshift(imageCCor)

row, col = image1.shape

yShift, xShift = np.unravel_index( np.argmax(imageCCorShift), (row,col) )

 



image1 = grid.regions[0].im0.pix 
image2 = grid.regions[0].im1.pix 


sx = 1024 ; ox = 10 
sy = 1024 ; oy = 10 

sx - sx*ox/100
sy - sy*oy/100 

ind1 = np.ix_( np.arange(image1.shape[0]), np.arange(   image1.shape[1] - int(np.ceil(sy*oy/100)), image1.shape[1] ) )
ind2 = np.ix_( np.arange(image2.shape[0]), np.arange(  int(np.ceil(sy*oy/100))  ) )

 
 

plt.figure()
plt.imshow(image1[ind1]) 

plt.figure()
plt.imshow(image2[ind2])

# Getting the shift between the croped images  

shift,_,_ = skimage.registration.phase_cross_correlation( image1[ind1], image2[ind2], upsample_factor=100 ) 

# Getting the global shift between the two images 

Tx = shift[0] 
Ty = image1.shape[1] - int(np.ceil(sy*oy/100)) + shift[1] 

image2_shift = sp.ndimage.shift(image2, [Tx,Ty])

 

plt.figure()
plt.imshow( image1 ) 

plt.figure()
plt.imshow( image2 )

plt.figure()
plt.imshow(image2_shift) , cmap = 'RdBu')
plt.colorbar()
plt.clim(-50,50)





plt.figure()
plt.imshow(np.real(image1FFT))


plt.figure()
plt.imshow(np.real(image2FFT))



translation(grid.regions[0].im0.pix[:, -100:],  grid.regions[0].im1.pix[:,:100 ]  )

 
result = ird.translation(grid.regions[0].im0.pix ,  grid.regions[0].im1.pix   )
tvec = result["tvec"].round(4)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)



import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 

# subpixel precision
#Upsample factor 100 = images will be registered to within 1/100th of a pixel.
#Default is 1 which means no upsampling.  
shifted, error, diffphase = register_translation(image, offset_image, 100)
print(f"Detected subpixel offset (y, x): {shifted}")

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(shifted[0], shifted[1]), mode='constant')
#plt.imshow(corrected_image)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()




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
plt.imshow(PxInvTot.reshape(X.shape)-X,cmap='RdBu')
cbar = plt.colorbar() 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20) 
plt.axis('equal')

plt.figure()
plt.imshow(PyInvTot.reshape(Y.shape)-Y,cmap='RdBu')
cbar = plt.colorbar() 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.axis('equal')

 
 
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

save_dire = '/media/rouwane/Crucial X6/_Pour Ali/DIC_3D_Confocal_prio/5-3_etap2/210820_162553/corrected_intensity/sigma=0.8/'

if not os.path.exists(save_dire):
   os.makedirs(save_dire)
   print("Saving directory "+save_dire+ "   created")

mx = np.array([3, 5, 6, 7]) 
my = np.array([3, 4, 6, 7]) 
sx, sy = 1024, 1024 
xc = sx/2 ; yc = sy/2 
 
# Step 0 parameters  
# p  = np.array([  ])
# cam = corrector.PolynomialCamera(p, xc, yc, mx, my)

x = np.arange(sx)
y = np.arange(sy)
X,Y = np.meshgrid(x,y,indexing='ij')
xtot = X.ravel() 
ytot = Y.ravel() 

Pxtot,Pytot = cam.P(xtot, ytot)

for i,fname in enumerate(os.listdir(dire)):
    print(i,fname)
    if fname[-3:] == 'tif': 
        im_new = corrector.Image(dire+fname)
        im_new.Load() 
        if im_new.pix.shape == (sx,sy): 
            im_new.BuildInterp()
            
            im_new.GaussianFilter(sigma=0.8)
         
            imc = im_new.Interp(Pxtot, Pytot)
            imc = imc.reshape((sx,sy)) 
            
            PILimg = PIL.Image.fromarray(np.round(imc).astype("uint8"))
            PILimg.save(save_dire+fname)  
        else: 
            print("Skip file")
    else: 
        print("Skip file")
    
    
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
 