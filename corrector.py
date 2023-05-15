import os
import numpy as np
import scipy as sp 
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import PIL.Image as image
import pathlib 
import cv2 
import skimage
from multiprocessing import Pool 
import concurrent.futures
import datetime 

  
    
class Region:
    """
    Overlapping region 
    It is defined between two images 
    """
    def __init__(self,im,index_im,band_type):
        self.im0  = im[0]  # Image of the reference configuration  
        self.im1  = im[1]  # Image of the deformed configuration 
        self.index_im0 = index_im[0] # Index of the reference image in the list of the images of the grid 
        self.index_im1 = index_im[1] # Index of the deformed image in the list of the images of the grid 
        
        self.type = band_type # Horizontal 'h' or vertical 'v' 

        
        self.xmin = None 
        self.xmax = None 
        self.ymin = None 
        self.ymax = None 
        
        self.bounds  = None  
        self.hx      = None 
        self.hy      = None   
        
        
        # [[xmin,ymin],[xmax,ymax]]
        # xmin = bounds[0,0]
        # xmax = bounds[1,0]
        # ymin = bounds[0,1]
        # ymax = bounds[1,1]
    
    def SetBounds(self, epsilon):
        self.xmin = int ( np.ceil(  max(min(self.im0.tx, self.im0.tx + self.im0.pix.shape[0]-1), min(self.im1.tx, self.im1.tx + self.im1.pix.shape[0]-1)) ) )
        self.ymin = int ( np.ceil(  max(min(self.im0.ty, self.im0.ty + self.im0.pix.shape[1]-1), min(self.im1.ty, self.im1.ty + self.im1.pix.shape[1]-1)) ) )
        self.xmax = int ( np.floor( min(max(self.im0.tx, self.im0.tx + self.im0.pix.shape[0]-1), max(self.im1.tx, self.im1.tx + self.im1.pix.shape[0]-1)) ) )
        self.ymax = int ( np.floor( min(max(self.im0.ty, self.im0.ty + self.im0.pix.shape[1]-1), max(self.im1.ty, self.im1.ty + self.im1.pix.shape[1]-1)) ) )
        
 
        self.xmin += epsilon 
        self.ymin += epsilon 
        self.xmax -= epsilon
        self.ymax -= epsilon
        
        self.hx = self.xmax - self.xmin 
        self.hy = self.ymax - self.ymin 
        
 
   
    def IntegrationPts(self, s = 1 ):
        x1d = np.arange( self.xmin, self.xmax )[::s]
        y1d = np.arange( self.ymin, self.ymax )[::s]
        X,Y = np.meshgrid(x1d,y1d,indexing='ij') 
        self.pgx = X.ravel() 
        self.pgy = Y.ravel() 
   
    def PlotResidual(self,res):
        shape = (self.xmax-self.xmin, self.ymax-self.ymin)
        
        plt.figure()
        plt.imshow(res.reshape(shape),cmap='RdBu')
 
    def PlotReferenceImage(self,cam=None):
        shape = (self.xmax-self.xmin,self.ymax-self.ymin)
        if cam is None:
            pgu,pgv = self.pgx - self.im0.tx, self.pgy - self.im0.ty
        else:
            pgu, pgv   = cam.P(self.pgx-self.im0.tx,self.pgy-self.im0.ty) 
       
        plt.figure()
        plt.imshow(self.im0.Interp(pgu,pgv).reshape(shape),cmap='gray')
      
 
    def PlotDeformedImage(self,cam=None):
        shape = (self.xmax-self.xmin,self.ymax-self.ymin)
        if cam is None:
            pgu,pgv = self.pgx-self.im1.tx,self.pgy-self.im1.ty
        else:
            pgu, pgv   = cam.P(self.pgx-self.im1.tx,self.pgy-self.im1.ty) 
        
        plt.figure()
        plt.imshow(self.im1.Interp(pgu,pgv).reshape(shape),cmap='gray')
 
    
    def Plot(self,color):
        """
        Plot of the region in the image domain 
        """
        plt.plot([self.ymin,self.ymax],[self.xmin,self.xmin], color=color)
        plt.plot([self.ymin,self.ymax],[self.xmax,self.xmax], color=color)
        plt.plot([self.ymin,self.ymin],[self.xmin,self.xmax], color=color)
        plt.plot([self.ymax,self.ymax],[self.xmin,self.xmax], color=color)
        
    def SetPairShift(self, cam, overlap):
        """
        Coputing the shift between the two neighboring images 
        of the current region 
        """
        
        # First getting two corrected croped regions 
        if self.type == 'v':
            x  =  np.arange(self.im0.pix.shape[0])
            yl =  np.arange(self.im1.pix.shape[1] - int(np.ceil(self.im1.pix.shape[1]*overlap[1]/100)), self.im1.pix.shape[1]) # Left points 
            yr =  np.arange( int(np.ceil(self.im1.pix.shape[1]*overlap[1]/100)) )  # Right points 
            
            X  = np.kron(x, np.ones(yl.size) ).reshape((-1,yl.size))
            Yl = np.kron(np.ones(x.size),yl).reshape((x.size,-1))
            Yr = np.kron(np.ones(x.size),yr).reshape((x.size,-1)) 
            
            pgul, pgvl = cam.P( X.ravel(), Yl.ravel() )
            pgur, pgvr = cam.P( X.ravel(), Yr.ravel() )
            
            # Left and right images 
            image1 = self.im0.Interp(pgul, pgvl).reshape(X.shape) 
            image2 = self.im1.Interp(pgur, pgvr).reshape(X.shape) 
            
            # Performing phase shift correlation 
            
            shift,_,_ = skimage.registration.phase_cross_correlation( image1, image2, upsample_factor=100 ) 
            print('Vertical overlap between image (%d,%d) and image (%d,%d)' % (self.im0.ix,self.im0.iy,self.im1.ix,self.im1.iy) )
            print(shift) 
             

            
            
        elif self.type == 'h':
            y = np.arange(self.im0.pix.shape[1]) 
            xt = np.arange(self.im1.pix.shape[0] - int(np.ceil(self.im1.pix.shape[0]*overlap[0]/100)), self.im1.pix.shape[0]) # Top points 
            xb = np.arange( int(np.ceil(self.im1.pix.shape[0]*overlap[0]/100)) )  # Bottom points  

            Y  = np.kron(np.ones(xt.size),y).reshape((xt.size,-1)) 
            Xt = np.kron(xt, np.ones(y.size) ).reshape((-1,y.size))
            Xb = np.kron(xb, np.ones(y.size) ).reshape((-1,y.size)) 
            
            pgut, pgvt = cam.P( Xt.ravel(), Y.ravel() )
            pgub, pgvb = cam.P( Xb.ravel(), Y.ravel() )
            
            # Top and bottom images 
            image1 = self.im0.Interp(pgut, pgvt).reshape(Y.shape) 
            image2 = self.im1.Interp(pgub, pgvb).reshape(Y.shape) 
            
            # Performing phase shift correlation 

            shift,_,_ = skimage.registration.phase_cross_correlation( image1, image2, upsample_factor=100 ) 
            
            print('Horizontal overlap between image (%d,%d) and image (%d,%d)' % (self.im0.ix,self.im0.iy,self.im1.ix,self.im1.iy) )
            print(shift)
 

 


        
    
    def GetOps(self,cam):
        
        # Assembles local operators 
        # defined on one region common between two images 
        # local DOF vector : [distortion,translation]
        # local DOF vector ; [a1,....,an,tx0,ty0,tx1,ty1]

        # Perform assembly of the DIC operators  
        
        # t1 = datetime.datetime.now()
        
        pgu1, pgv1   = cam.P(self.pgx-self.im0.tx,self.pgy-self.im0.ty) 
        pgu2, pgv2   = cam.P(self.pgx-self.im1.tx,self.pgy-self.im1.ty)  
        f1p            = self.im0.Interp(pgu1, pgv1)
        df1dx, df1dy   = self.im0.InterpGrad(pgu1,pgv1)        
        f2p            = self.im1.Interp(pgu2, pgv2)
        df2dx, df2dy   = self.im1.InterpGrad(pgu2,pgv2)        
                
        n = len(self.pgx)
    
        Jpu1, Jpv1   = cam.dPdp(self.pgx-self.im0.tx,self.pgy-self.im0.ty) 
        dudx, dudy, dvdx, dvdy = cam.dPdX(self.pgx-self.im0.tx,self.pgy-self.im0.ty)  
        JXu1       = np.zeros((n,4))
        JXv1       = np.zeros((n,4))
        JXu1[:,0]  = -dudx  ; JXu1[:,1]  = -dudy 
        JXv1[:,0]  = -dvdx  ; JXv1[:,1]  = -dvdy    
        
        Jpu1 = np.hstack((Jpu1,JXu1))
        Jpv1 = np.hstack((Jpv1,JXv1))
        
        Jpu2, Jpv2   = cam.dPdp(self.pgx-self.im1.tx,self.pgy-self.im1.ty) 
        dudx, dudy, dvdx, dvdy = cam.dPdX(self.pgx-self.im1.tx,self.pgy-self.im1.ty)
        JXu2       = np.zeros((n,4))
        JXv2       = np.zeros((n,4))
        JXu2[:,2]  = -dudx   ; JXu2[:,3]  = -dudy 
        JXv2[:,2]  = -dvdx   ; JXv2[:,3]  = -dvdy            
        Jpu2 = np.hstack((Jpu2, JXu2))
        Jpv2 = np.hstack((Jpv2, JXv2))
        
        Jp1  = np.vstack((Jpu1,Jpv1))
        Jp2  = np.vstack((Jpu2,Jpv2))
        
        df1  =  sp.sparse.dia_matrix((np.vstack((df1dx, df1dy)), np.array([0,-n])), shape=(2*n,n))
        df2  =  sp.sparse.dia_matrix((np.vstack((df2dx, df2dy)), np.array([0,-n])), shape=(2*n,n))      
        M    =  df1.T.dot(Jp1) - df2.T.dot(Jp2)   
        H    =  M.T.dot(M) 
        # res  =  f1p - f2p 
        res  =  f1p - np.mean(f1p) - (f2p - np.mean(f2p) ) # Offset brightness correction 
        
        b    =  -M.T.dot(res) 
    
        return H,b,res           
            
    # def RunCorrelation(p0,cam,Niter):
        
            

class Grid:
    """
    Totality of regions 
    Related to the distorted images to be stitched 
    """
    def __init__(self,nImages,overlap,shape,images,regions):
        """
        shape : (nx,ny) : number of images in x and y directions 
        overlap : (ox,oy): size of the overlap in x and y directions in % 
        imfiles : list of the image files 
        """
        self.sx = shape[0]
        self.sy = shape[1]
        self.images = images 
        self.regions = regions 
        self.nx = nImages[0]
        self.ny = nImages[1]
        self.ox = overlap[0]
        self.oy = overlap[1] 
    
    def BuildInterp(self, method = 'cubic-spline'):
        for im in self.images:
            im.BuildInterp(method)
    def GaussianFilter(self, sigma):
        for im in self.images:
            im.GaussianFilter(sigma) 
    def LoadImages(self):
        for im in self.images:
            im.Load() 
    def Connectivity(self,conn):
        self.conn = conn 
            
    
    def ExportTile(self, file):
        """ Writing Tile Configuration file for FIJI 
            Warning !! Inverted x and y axis for FIJI """ 
        with open(file, 'w') as f:
            f.write('# Define the number of dimensions we are working on\n')
            f.write('dim = 2\n')
            f.write('\n# Define the image coordinates\n')  
            for im in self.images:
                f.write(pathlib.Path(im.fname).name+"; ; "+"(%f,%f)"%(im.ty,im.tx)+"\n")


    def ReadTile(self, file):
        """ Reading Tile configuration File from FIJI  
            Warning !! Inverted x and y axis for FIJI """      
        im_file_names = [pathlib.Path(im.fname).name for im in self.images ] 
        tile = open(file)  
        line = tile.readline()
        while line!='# Define the image coordinates\n':
            line = tile.readline()
        line = tile.readline()        
        while len(line)!=0 : 
            linec  = line.split(';')
            # Reading image 
            im_file_name   = linec[0]
            coords = linec[2][2:-2].split(',')
            tx = float(coords[1])
            ty = float(coords[0])
            index_im  = im_file_names.index(im_file_name)
            self.images[index_im].SetCoordinates(tx,ty)
            line   = tile.readline()
 
    
    def PlotImageBoxes(self, origin=(0,0),eps=(0,0)):
        # A = np.zeros( ( int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0], 
        #                 int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1]  ) )
        # plt.imshow(A)
        for im in self.images:
            im.PlotBox(origin,color='green')
    
    def StitchImages(self, cam, origin=(0,0), eps=(0,0), fusion_mode='average'):
        for r in self.regions: 
            r.SetBounds(epsilon=0) 
        if fusion_mode == 'average': 
            # Size of the fused image 
            sx_ims = int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0]
            sy_ims = int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1] 
            # Discretize all the image domain 
            X,Y = np.meshgrid( np.arange(sx_ims),  np.arange(sy_ims), indexing='ij' )
            x = X.ravel() 
            y = Y.ravel() 
            ns = len(x)
            # Creating empty image and the average weights 
            ims = np.zeros(sx_ims*sy_ims)
            ws = np.zeros(ns)
            print('Fusing images (average):')
            for k, im in enumerate(self.images):
                ProgressBar(100 * (k+1) / len(self.images))
                # print(k,end=',')
                
                # Restrict the FOV test only on a region (for faster calculation)
                windowExt = [10,10]
                imin  =  np.maximum(0, np.floor(im.tx+origin[0]-windowExt[0]) ) 
                imax  =  np.minimum(sx_ims-1 , np.ceil(im.tx+origin[0]+im.pix.shape[0]+windowExt[0])  )
                jmin  =  np.maximum(0, np.floor(im.ty+origin[1]-windowExt[1]) )
                jmax  =  np.minimum(sy_ims-1 , np.ceil(im.ty+origin[1]+im.pix.shape[1]+windowExt[1]) ) 
                I,J = np.meshgrid(np.arange(imin,imax+1),np.arange(jmin,jmax+1),indexing='ij')
                indexI = I.ravel() 
                indexJ = J.ravel() 
                index1 = (indexJ + indexI*X.shape[1]).astype('int')
                # Add the contribution of the sub-image to the total image 
                u,v      = cam.P( x[index1]-origin[0]-im.tx, y[index1]-origin[1]-im.ty )
                fov1 =  ((u >= 0)*(u<= im.pix.shape[0]-1 )*(v >= 0)*(v<=im.pix.shape[1]-1)).astype('int')
                index2   = np.where(fov1==1)[0] # Points that are in the field of view 
                fov      = index1[index2] 
                ws[fov]  += 1
                ims[fov] += im.Interp(u[index2], v[index2])   # Image 
            #mask = np.copy(ws); mask[mask>0] = 1
            ws[ws==0] = 1
            ims = ims/ws 
            print('\n')
            return ims.reshape((sx_ims,sy_ims)) 
        
        if fusion_mode == 'linear blending':
            # Size of the fused image 
            sx_ims = int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0]
            sy_ims = int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1] 
            # Discretize all the image domain 
            X,Y = np.meshgrid( np.arange(sx_ims),  np.arange(sy_ims), indexing='ij' )
            x = X.ravel() 
            y = Y.ravel() 
            ns = len(x)
            # Creating empty image and the average weights 
            ims = np.zeros(sx_ims*sy_ims)
            ws  = np.zeros(ns)
            print('Fusing images (linear blending):')
            for k, r in enumerate(self.regions):
                ProgressBar(100 * (k+1) / len(self.regions))
                if r.type == 'v':
                    """ Left image im0 """ 
                    windowExt = [10,10]
                    imin  =  np.maximum(0, np.floor(r.im0.tx+origin[0]-windowExt[0]) ) 
                    imax  =  np.minimum(sx_ims-1 , np.ceil(r.im0.tx + origin[0] + self.sx + windowExt[0])  )
                    jmin  =  np.maximum(0, np.floor(r.im0.ty + origin[1] - windowExt[1]) )
                    jmax  =  np.minimum(sy_ims-1 , np.ceil(r.im0.ty + origin[1] + self.sy + windowExt[1]) ) 
                    I,J = np.meshgrid(np.arange(imin,imax+1),np.arange(jmin,jmax+1),indexing='ij')
                    indexI = I.ravel() 
                    indexJ = J.ravel() 
                    index0 = (indexJ + indexI*X.shape[1]).astype('int')
                    u, v        = cam.P( x[index0]-origin[0]-r.im0.tx, y[index0]-origin[1]-r.im0.ty )
                    fov1        =  ((u >= 0)*(u<= self.sx-1 )*(v >= 0)*(v<=self.sy-1)).astype('int')
                    index1      = np.where(fov1==1)[0]
                    fov2        = ((v[index1]>=0)*(v[index1]<=self.sy-1-r.hy)).astype('int')
                    index2_o    = np.where(fov2==0)[0]  # Overlapping region  
                    index2_no   = np.where(fov2==1)[0]  # Non overlapping region  
                    fov_o       =  index1[index2_o] 
                    fov_no      =  index1[index2_no] 
                    w           =  ( v[fov_o] - (self.sy-1) ) / (-r.hy)  # Decreasing weighting 
                    ws[index0[fov_no]]  +=  1
                    ims[index0[fov_no]] +=  r.im0.Interp(u[fov_no],v[fov_no]) 
                    ws[index0[fov_o]]   +=  w 
                    ims[index0[fov_o]]  +=  r.im0.Interp(u[fov_o],v[fov_o]) * w 
                    
                    # plt.figure() 
                    # plt.imshow(r.im0.pix, cmap='gray')
                    # plt.plot(v[fov_no], u[fov_no], '.', markersize=2, color='red')
                    # plt.plot(v[fov_o], u[fov_o], '.', markersize=2, color='green')
                    
                    # ws[ws==0] = 1
                    # ims = ims/ws
                    # plt.figure()
                    # plt.imshow( ims.reshape((sx_ims,sy_ims)), cmap='gray' )
 
                    # raise ValueError('Stop debug')
                    
                    """ Right image  im1"""  
                    windowExt = [10,10]
                    imin  =  np.maximum(0, np.floor(r.im1.tx+origin[0]-windowExt[0]) ) 
                    imax  =  np.minimum(sx_ims-1 , np.ceil(r.im1.tx + origin[0] + self.sx + windowExt[0])  )
                    jmin  =  np.maximum(0, np.floor(r.im1.ty + origin[1] - windowExt[1]) )
                    jmax  =  np.minimum(sy_ims-1 , np.ceil(r.im1.ty + origin[1] + self.sy + windowExt[1]) ) 
                    I,J = np.meshgrid(np.arange(imin,imax+1),np.arange(jmin,jmax+1),indexing='ij')
                    indexI = I.ravel() 
                    indexJ = J.ravel() 
                    index0 = (indexJ + indexI*X.shape[1]).astype('int')
                    u, v   = cam.P( x[index0]-origin[0]-r.im1.tx, y[index0]-origin[1]-r.im1.ty )
                    fov1   =  ((u >= 0)*(u<= self.sx-1 )*(v >= 0)*(v<=self.sy-1)).astype('int')
                    index1 = np.where(fov1==1)[0]
                    fov2   = ((v[index1]>=0)*(v[index1]<=r.hy)).astype('int')
                    index2_o  = np.where(fov2==1)[0] # Overlapping region  
                    index2_no = np.where(fov2==0)[0] # Non overlapping region 
                    fov_o       =  index1[index2_o] 
                    fov_no      =  index1[index2_no] 
                    w           =  ( v[fov_o] ) / (r.hy)     # Increasing weighting 
                    ws[index0[fov_no]]  +=  1
                    ims[index0[fov_no]] +=  r.im1.Interp(u[fov_no],v[fov_no]) 
                    ws[index0[fov_o]]   +=  w 
                    ims[index0[fov_o]]  +=  r.im1.Interp(u[fov_o],v[fov_o]) * w 
                    
                    
                    # plt.figure() 
                    # plt.imshow(r.im1.pix, cmap='gray')
                    # plt.plot(v[fov_no], u[fov_no], '.', markersize=2, color='red')
                    # plt.plot(v[fov_o], u[fov_o], '.', markersize=2, color='green')
                    
                    # ws[ws==0] = 1
                    # ims = ims/ws
                    # plt.figure()
                    # plt.imshow( ims.reshape((sx_ims,sy_ims)), cmap='gray' )
                    
                    # raise ValueError('Stop debug')
                    
                    
                elif r.type == 'h':
                    """ Top image im0 """ 
                    windowExt = [10,10]
                    imin  =  np.maximum(0, np.floor(r.im0.tx+origin[0]-windowExt[0]) ) 
                    imax  =  np.minimum(sx_ims-1 , np.ceil(r.im0.tx + origin[0] + self.sx + windowExt[0])  )
                    jmin  =  np.maximum(0, np.floor(r.im0.ty + origin[1] - windowExt[1]) )
                    jmax  =  np.minimum(sy_ims-1 , np.ceil(r.im0.ty + origin[1] + self.sy + windowExt[1]) ) 
                    I,J = np.meshgrid(np.arange(imin,imax+1),np.arange(jmin,jmax+1),indexing='ij')
                    indexI = I.ravel() 
                    indexJ = J.ravel() 
                    index0 = (indexJ + indexI*X.shape[1]).astype('int')
                    u, v       = cam.P( x[index0]-origin[0]-r.im0.tx, y[index0]-origin[1]-r.im0.ty )
                    fov1       =  ((u >= 0)*(u<= self.sx-1 )*(v >= 0)*(v<=self.sy-1)).astype('int')
                    index1      = np.where(fov1==1)[0]
                    fov2        = ((u[index1]>=0)*(u[index1]<=self.sx-1-r.hx)).astype('int')
                    index2_o    = np.where(fov2==0)[0]  # Overlapping region  
                    index2_no   = np.where(fov2==1)[0]  # Non overlapping region  
                    fov_o       =  index1[index2_o] 
                    fov_no      =  index1[index2_no] 
                    w           =  ( u[fov_o] - (self.sx-1) ) / (-r.hx)  # Decreasing weighting 
                    ws[index0[fov_no]]  +=  1
                    ims[index0[fov_no]] +=  r.im0.Interp(u[fov_no],v[fov_no]) 
                    ws[index0[fov_o]]   +=  w 
                    ims[index0[fov_o]]  +=  r.im0.Interp(u[fov_o],v[fov_o]) * w 
                    """ Bottom image im1"""  
                    windowExt = [10,10]
                    imin  =  np.maximum(0, np.floor(r.im1.tx+origin[0]-windowExt[0]) ) 
                    imax  =  np.minimum(sx_ims-1 , np.ceil(r.im1.tx + origin[0] + self.sx + windowExt[0])  )
                    jmin  =  np.maximum(0, np.floor(r.im1.ty + origin[1] - windowExt[1]) )
                    jmax  =  np.minimum(sy_ims-1 , np.ceil(r.im1.ty + origin[1] + self.sy + windowExt[1]) ) 
                    I,J = np.meshgrid(np.arange(imin,imax+1),np.arange(jmin,jmax+1),indexing='ij')
                    indexI = I.ravel() 
                    indexJ = J.ravel() 
                    index0 = (indexJ + indexI*X.shape[1]).astype('int')
                    u, v   = cam.P( x[index0]-origin[0]-r.im1.tx, y[index0]-origin[1]-r.im1.ty )
                    fov1   =  ((u >= 0)*(u<= self.sx-1 )*(v >= 0)*(v<=self.sy-1)).astype('int')
                    index1 = np.where(fov1==1)[0]
                    fov2   = ((u[index1]>=0)*(u[index1]<=r.hx)).astype('int')
                    index2_o  = np.where(fov2==1)[0] # Overlapping region  
                    index2_no = np.where(fov2==0)[0] # Non overlapping region 
                    fov_o       =  index1[index2_o] 
                    fov_no      =  index1[index2_no] 
                    w           =  ( u[fov_o] ) / (r.hx)     # Increasing weighting 
                    ws[index0[fov_no]]  +=  1
                    ims[index0[fov_no]] +=  r.im1.Interp(u[fov_no],v[fov_no]) 
                    ws[index0[fov_o]]   +=  w 
                    ims[index0[fov_o]]  +=  r.im1.Interp(u[fov_o],v[fov_o]) * w 
                else:
                    raise ValueError('Unknown region type')
            ws[ws==0] = 1
            ims = ims/ws 
            print('\n')
            # plt.figure()
            # plt.imshow(ws.reshape(sx_ims,sy_ims))
            return ims.reshape((sx_ims,sy_ims)) 

        else:
            raise ValueError('Unknown blending method')
                
                
            
            
            
            
        # Map each image in the reference coordinate system related to the first 
        # image with the offset origin 
        
 
        
        # for k, im in enumerate(self.images):
        #     print('Image '+str(k))
            
        #     xmin = im.tx  + origin[0] ;  xmax =  im.tx + im.pix.shape[0] + origin[0]
        #     ymin = im.ty  + origin[1] ;  ymax =  im.ty + im.pix.shape[1] + origin[1]
            
        #     xmin = int(np.round(xmin)) ;  xmax = int(np.round(xmax))
        #     ymin = int(np.round(ymin)) ;  ymax = int(np.round(ymax))
            
        #     X,Y = np.meshgrid( np.arange( xmin, xmax ) , np.arange( ymin, ymax ) ,indexing='ij')
        #     i = X.ravel() 
        #     j = Y.ravel() 

        #     Xim,Yim = np.meshgrid(np.arange(self.sx),np.arange(self.sy),indexing='ij')
        #     u = Xim.ravel() 
        #     v = Yim.ravel() 
        #     Px,Py = cam.P(u,v)
            
        #     ims[i,j] =  im.Interp(Px,Py)
        
        # return ims 
    
    # def GetOpsParallel(self,cam):
        
    #     def local_assembly(r,cam):
    #         return r.GetOps(cam)
        
    #     with Pool() as pool:
    #         results = pool.map(local_assembly, [(self.regions[i],cam) for i in range(len((self.regions))) ] ) 
    #         print('Results')
    #         print(results)
    #         for r in results:
    #             print(r.get())
    #         # for H,b,res in results: 
    #         #     # print(H,b,res)
    #         #     print('')
        
    #     return 
    
    def local_assembly(self,r,cam):
        """
        Redefined function for the parallelization 
        """
        return r.GetOps(cam)
    
    def GetOps(self,cam):
        # t1 = datetime.datetime.now() 
        nd   = len(cam.p)  # Number of distortion parameters 
        nImages = len(self.images)
        ndof = nd + 2*nImages
        H   = np.zeros((ndof,ndof))
        b   = np.zeros(ndof)
        res_tot = [None]*len(self.regions)
        arnd = np.arange(nd)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor :
            processes = [ executor.submit(self.local_assembly, self.regions[i], cam) for i in range(len((self.regions)))  ]
        for i, p in enumerate(processes):
            r = self.regions[i]
            Hl,bl,resl =   p.result()
            rep  =  np.concatenate((arnd, self.conn[r.index_im0,:], self.conn[r.index_im1,:])) 
            repk =  np.ix_(rep,rep)
            H[repk] +=  Hl
            b[rep]  +=  bl   
            res_tot[i]  = resl
        # print( "Time ", datetime.datetime.now() - t1 )  
        return H,b,res_tot   
    
    # def GetOps(self,cam):
    #     t1 = datetime.datetime.now() 

    #     # Global DOF vector ; [a1,....,an,tx0,ty0,tx1,ty1,...,txn,tyn]
    #     nd   = len(cam.p)  # Number of distortion parameters 
    #     nImages = len(self.images)
    #     ndof = nd + 2*nImages
    #     H   = np.zeros((ndof,ndof))
    #     b   = np.zeros(ndof)
    #     res_tot = [None]*len(self.regions)
    #     arnd = np.arange(nd)
    #     for i,r in enumerate(self.regions):
    #         Hl,bl,resl =   r.GetOps(cam)
    #         rep  =  np.concatenate((arnd, self.conn[r.index_im0,:], self.conn[r.index_im1,:])) 
    #         repk =  np.ix_(rep,rep)
    #         H[repk] +=  Hl
    #         b[rep]  +=  bl   
    #         res_tot[i]  = resl
    #     print( "Time ", datetime.datetime.now() - t1 )  
    #     return H,b,res_tot   
    
    def PlotResidualMap(self,res_list, epsImage = 10):
        # Size of the fused image 
        sx_ims = int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100) + epsImage )
        sy_ims = int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100) + epsImage )  
        # Creating empty image and the average weights 
        R = np.zeros((sx_ims,sy_ims)) 
        for i,r in enumerate(self.regions):
            shape = (r.xmax - r.xmin, r.ymax - r.ymin)
            X = r.pgx.reshape(shape)
            Y = r.pgy.reshape(shape)
            R[X,Y] = res_list[i].reshape(shape)
        plt.imshow(R,cmap='RdBu')

 
    
    def RunGN(self,cam):
        return 
    
    def SetPairShift(self,cam,overlap):
        for r in self.regions:
            r.SetPairShift(cam, overlap) 
            
            
            



class Image:
    def __init__(self, fname):
        """Contructor"""
        self.fname = fname
        self.tx = None 
        self.ty = None 
        self.ix = None 
        self.iy = None 
        self.txr = None # Relative shift to upper image  
        self.tyr = None # Relative shift to left image  
        
    
    def SetCoordinates(self,tx,ty):
        self.tx = tx 
        self.ty = ty 
    
    def SetHorizontalShift(self,t):
        self.tyr = t 
    
    def SetVerticalShift(self,t):
        self.txr = t 
        
    
    
    def SetIndices(self,ix,iy):
        self.ix = ix 
        self.iy = iy 
    
    def PlotBox(self,origin, color):
        
        xmin = self.tx  + origin[0] ;  xmax =  self.tx + self.pix.shape[0] + origin[0]
        ymin = self.ty  + origin[1] ;  ymax =  self.ty + self.pix.shape[1] + origin[1]
        
        plt.plot([ymin,ymax],[xmin,xmin], color=color)
        plt.plot([ymin,ymax],[xmax,xmax], color=color)
        plt.plot([ymin,ymin],[xmin,xmax], color=color)
        plt.plot([ymax,ymax],[xmin,xmax], color=color)
 
    def Load(self):
        """Load image data"""
        if os.path.isfile(self.fname):
            if self.fname.split(".")[-1] == "npy":
                self.pix = np.load(self.fname)
            else:
                self.pix = np.asarray(image.open(self.fname)).astype(float)
                # self.pix = image.imread(self.fname).astype(float)
            if len(self.pix.shape) == 3:
                self.ToGray()
        else:
            print("File "+self.fname+" not in directory "+os.getcwd())
        return self

    def Load_cv2(self):
        """Load image data using OpenCV"""
        if os.path.isfile(self.fname):
            import cv2 as cv
            self.pix = cv.imread(self.fname).astype(float)
            if len(self.pix.shape) == 3:
                self.ToGray()
        else:
            print("File "+self.fname+" not in directory "+os.getcwd())
        return self

    def Copy(self):
        """Image Copy"""
        newimg = Image("Copy")
        newimg.pix = self.pix.copy()
        return newimg

    def Save(self, fname):
        """Image Save"""
        PILimg = image.fromarray(np.round(self.pix).astype("uint8"))
        PILimg.save(fname)
        # image.imsave(fname,self.pix.astype('uint8'),vmin=0,vmax=255,format='tif')
        
        
    def BuildInterp(self,method='cubic-spline'):
        x = np.arange(0, self.pix.shape[0])
        y = np.arange(0, self.pix.shape[1])
        self.interpMethod = method 
        if method == 'cubic-spline':
            self.tck = spi.RectBivariateSpline(x, y, self.pix, kx=3, ky=3)
        if method == 'linear':
            self.interp = spi.RegularGridInterpolator((x,y), self.pix, method='linear', bounds_error=False, fill_value=None)

            # Computing image gradients
            # self.dpixdx = cv2.Sobel(self.pix,cv2.CV_64F,1,0,ksize=-1)
            # self.dpixdy = cv2.Sobel(self.pix,cv2.CV_64F,0,1,ksize=-1)
            
            self.dpixdx = np.gradient(self.pix, axis=0)
            self.dpixdy = np.gradient(self.pix, axis=1)

            self.interp_dpixdx = spi.RegularGridInterpolator((x,y), self.dpixdx, method='nearest',bounds_error = False , fill_value=None)
            self.interp_dpixdy = spi.RegularGridInterpolator((x,y), self.dpixdy, method='nearest',bounds_error = False , fill_value=None)
    
    def Interp(self, x, y):
        """evaluate interpolator at non-integer pixel position x, y"""
        if self.interpMethod == 'cubic-spline':    
            return self.tck.ev(x, y)
        if self.interpMethod == 'linear':
            return self.interp(np.vstack((x,y)).T)
        # if self.interpMethod == 'nearest':
        #     # x[x<0] = 0 
        #     # y[y<0] = 0 
        #     # x[x>self.pix.shape[0]-1] = self.pix.shape[0]-1
        #     # y[y>self.pix.shape[1]-1] = self.pix.shape[1]-1
        #     # print(np.min(x),np.max(x),np.min(y),np.max(y))
        #     # print(self.pix.shape[0]-1,self.pix.shape[1]-1)
        #     return self.interp_pix(np.vstack((x,y)).T)
       
    
    def InterpGrad(self, x, y, eps=1.e-7):
        """evaluate gradient of the interpolator at non-integer pixel position x, y"""
        if self.interpMethod == 'cubic-spline': 
            return self.tck.ev(x, y, 1, 0), self.tck.ev(x, y, 0, 1) 
        # if self.interpMethod == 'linear':
        #     X = np.vstack((x,y)).T 
        #     Xx1 = X.copy()  ; Xx1[:,0] += eps/2
        #     Xx2 = X.copy()  ; Xx2[:,0] -= eps/2 
        #     Xy1 = X.copy()  ; Xy1[:,1] += eps/2 
        #     Xy2 = X.copy()  ; Xy2[:,1] -= eps/2  
        #     gx =  ( self.interp( Xx1 ) - self.interp( Xx2 )) / eps 
        #     gy =  ( self.interp( Xy1 ) - self.interp( Xy2 )) / eps 
        #     return gx,gy
        if self.interpMethod == 'linear':
            gx =   self.interp_dpixdx(np.vstack((x,y)).T)
            gy =   self.interp_dpixdy(np.vstack((x,y)).T)
            return gx,gy
        
        # if self.interpMethod=='nearest':
        #     pts = np.vstack((x,y)).T
        #     return self.interp_dpixdx(pts), self.interp_dpixdy(pts)
 

    def Plot(self):
        """Plot Image"""
        plt.imshow(self.pix, cmap="gray", interpolation="none", origin="upper")
        # plt.axis('off')
        # plt.colorbar()

    def Dynamic(self):
        """Compute image dynamic"""
        g = self.pix.ravel()
        return max(g) - min(g)

    def GaussianFilter(self, sigma=0.7):
        """Performs a Gaussian filter on image data. 

        Parameters
        ----------
        sigma : float
            variance of the Gauss filter."""
        from scipy.ndimage import gaussian_filter

        self.pix = gaussian_filter(self.pix, sigma)

    def PlotHistogram(self):
        """Plot Histogram of graylevels"""
        plt.hist(self.pix.ravel(), bins=125, range=(0.0, 255), fc="k", ec="k")
        plt.show()

    def SubSample(self, n):
        """Image copy with subsampling for multiscale initialization"""
        scale = 2 ** n
        sizeim1 = np.array([self.pix.shape[0] // scale, self.pix.shape[1] // scale])
        nn = scale * sizeim1
        im0 = np.mean(
            self.pix[0 : nn[0], 0 : nn[1]].T.reshape(np.prod(nn) // scale, scale),
            axis=1,
        )
        nn[0] = nn[0] // scale
        im0 = np.mean(
            im0.reshape(nn[1], nn[0]).T.reshape(np.prod(nn) // scale, scale), axis=1
        )
        nn[1] = nn[1] // scale
        self.pix = im0.reshape(nn)

    def ToGray(self, type="lum"):
        """Convert RVG to Grayscale :

        Parameters
        ----------
        type : string
            lig : lightness
            lum : luminosity (DEFAULT)
            avg : average"""
        if type == "lum":
            self.pix = (
                0.21 * self.pix[:, :, 0]
                + 0.72 * self.pix[:, :, 1]
                + 0.07 * self.pix[:, :, 2]
            )
        elif type == "lig":
            self.pix = 0.5 * np.maximum(
                np.maximum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            ) + 0.5 * np.minimum(
                np.minimum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            )
        else:
            self.pix = np.mean(self.pix, axis=2)

    def SelectPoints(self, n=-1, title=None):
        """Select a point in the image. 
        
        Parameters
        ----------
        n : int
            number of expected points
        title : string (OPTIONNAL)
            modify the title of the figure when clic is required.
            
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if title is None:
            if n < 0:
                plt.title("Select some points... and press enter")
            else:
                plt.title("Select " + str(n) + " points... and press enter")
        else:
            plt.title(title)
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        return pts1
 

class PolynomialCamera:
    """
    Polynomial camera model 
    constant, x, y, x*y, x**2, y**2, x**2*y, x*y**2, x**3, y**3 
    """
    def __init__(self, p, xc,yc, mx,my):
        self.p  = p 
        self.mx = mx 
        self.my = my 
        self.xc = xc 
        self.yc = yc 
        
    def P(self,X,Y): 
 
        x = X - self.xc 
        y = Y - self.yc 
        A = np.vstack((np.ones(len(x)),x,y,x*y,x**2,y**2,x**2*y,x*y**2,x**3,y**3)).T 
        
        Ax = A[:,self.mx]
        Ay = A[:,self.my]
        
        n = len(self.mx)
        px = self.p[:n]
        py = self.p[n:]
        u = X + Ax.dot(px)
        v = Y + Ay.dot(py)
        
        # Prendre en compte le cas ou zero distortion 
        # u = X 
        # v = Y 
        
        
        return u,v 
    
    def dPdX(self,X,Y):
        zero = np.zeros(len(X))
        one  = np.ones(len(X))
        
        x = X - self.xc
        y = Y - self.yc
        dAdX = np.vstack((zero,one,zero,y,2*x,zero,2*x*y,y**2,3*x**2,zero)).T
        dAdY = np.vstack((zero,zero,one,x,zero,2*y,x**2,2*x*y,zero,3*y**2)).T 
        
        dAxdX = dAdX[:,self.mx]
        dAxdY = dAdY[:,self.mx]
        dAydX = dAdX[:,self.my]
        dAydY = dAdY[:,self.my]
        
        n = len(self.mx)
        px = self.p[:n]
        py = self.p[n:] 
        dudx = 1 + dAxdX.dot(px)
        dudy = dAxdY.dot(px)
        dvdx = dAydX.dot(py)
        dvdy = 1 + dAydY.dot(py)
        return dudx, dudy, dvdx, dvdy
    
    def dPdp(self,X,Y): 
        x = X - self.xc 
        y = Y - self.yc
        A = np.vstack((np.ones(len(x)),x,y,x*y,x**2,y**2,x**2*y,x*y**2,x**3,y**3)).T
        
        Ax = A[:,self.mx]
        Ay = A[:,self.my] 
        return np.hstack((Ax, np.zeros((len(X),len(self.mx))))), np.hstack((np.zeros((len(X),len(self.my))), Ay)) 
    
    def Pinv(self,U,V):
        pu,pv = self.P(U, V)
        Xold = U - pu 
        Yold = V - pv 
        
        error = 1  
        while error>1.e-8: 
            X,Y = self.P(Xold,Yold)
            dPxdx, dPxdy, dPydx, dPydy = self.dPdX(Xold,Yold)
            detJac =  dPxdx*dPydy - dPydx*dPxdy 
            Xnew = Xold + (1./detJac)*(  dPydy*(U-X) - dPxdy*(V-Y) )
            Ynew = Yold + (1./detJac)*( -dPydx*(U-X) + dPxdx*(V-Y) )
            error =  max( np.max( np.abs( Xnew - Xold )) , np.max( np.abs( Ynew - Yold )) ) 
            Xold = Xnew 
            Yold = Ynew  
            print('--Error ', error) 
        return Xold,Yold 
    
    def Plot(self, size):
        x = np.arange(size[0])
        y = np.arange(size[1])
        X,Y = np.meshgrid(x,y,indexing='ij')
        xtot = X.ravel() 
        ytot = Y.ravel() 
    
        Pxtot,Pytot = self.P(xtot, ytot)
    
        plt.figure()
        plt.imshow(Pxtot.reshape(X.shape)-X,cmap='RdBu')
        cbar = plt.colorbar() 
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cbar.ax.tick_params(labelsize=20) 
        plt.axis('equal')
        
        plt.figure()
        plt.imshow(Pytot.reshape(Y.shape)-Y,cmap='RdBu')
        cbar = plt.colorbar() 
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.axis('equal')
    
    def PinvGrid(self, size , alpha):
        
        roi = np.array([[0,0],[size[0],size[1]]])   # Plot (y,x) in inverted axis (image system)
        m,_= MeshFromROI(roi, dx=60)

        xn1, yn1  = m.n[:,0], m.n[:,1]
        px, py = self.Pinv(m.n[:,0] , m.n[:,1]  )
        
        plt.figure(figsize=(10,10))
        plt.ylim(-100,1000)
        plt.xlim(100,200)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        m.Plot(n=np.c_[yn1,xn1]   , edgecolor='red', alpha=1)
        m.Plot(n=np.c_[yn1+(py-yn1)*alpha,xn1+(px-xn1)*alpha]  , edgecolor='blue', alpha=1)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        
        
        
    
    def PlotInverse(self,size):
        x = np.arange(size[0])
        y = np.arange(size[1])
        X,Y = np.meshgrid(x,y,indexing='ij')
        xtot = X.ravel() 
        ytot = Y.ravel() 
    
        PxInvTot, PyInvTot = self.Pinv(xtot, ytot)
    
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



class ParametricCameraBis:
    """
    Brown-Conrady parametric distortion model 
    """ 
    def __init__(self, p, xc, yc):
        self.p  = p 
        self.xc = xc 
        self.yc = yc 

    def P(self,X,Y):
        # rho2 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**1 
        # rho4 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**2 
        # rho6 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**3 
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2  

        u = X + (X - self.xc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 ) + (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*(1 + self.p[5]*r2 + self.p[6]*r4)
        v = Y + (Y - self.yc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 ) + (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*(1 + self.p[5]*r2 + self.p[6]*r4)   
        
        return u,v 
    
    def dPdX(self,X,Y):
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2 
        
        K1,K2,K3,P1,P2,P3,P4 = self.p
 
        
        dudx = 1 + K1*r2 + K2*r4 + K3*r6 + (X - self.xc)**2*(2*K1 + 4*K2*r2 + 6*K3*r4) +\
               (6*P1*(X - self.xc) + 2*P2*(Y - self.yc))*(1 + P3*r2 + P4*r4) +\
               (P1*(r2 + 2*(X - self.xc)**2) + 2*P2*(X - self.xc)*(Y - self.yc))*(2*P3*(X - self.xc) + 4*P4*(X - self.xc)*r2)    
        
        dudy = (X - self.xc)*(Y - self.yc)*(2*K1 + 4*K2*r2 + 6*K3*r4) + (2*P1*(Y - self.yc) + 2*P2*(X - self.xc))*(1 + P3*r2 + P4*r4) +\
               (P1*(r2 + 2*(X - self.xc)**2) + 2*P2*(X - self.xc)*(Y - self.yc))*(2*P3*(Y - self.yc) + 4*P4*(Y - self.yc)*r2)
        
        dvdx =  dudy 
        
        dvdy =  1 + K1*r2 + K2*r4 + K3*r6 + (Y - self.yc)**2*(2*K1 + 4*K2*r2 + 6*K3*r4)   +\
                (6*P1*(Y - self.yc) + 2*P2*(X - self.xc))*(1 + P3*r2 + P4*r4) +\
                (P1*(r2 + 2*(Y - self.yc)**2) + 2*P2*(Y - self.yc)*(X - self.xc))*(2*P3*(Y - self.yc) + 4*P4*(Y - self.yc)*r2)

        
        return  dudx, dudy, dvdx, dvdy 
    
    def dPdp(self,X,Y):
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2 
        
        dudK1 = (X - self.xc)*r2
        dudK2 = (X - self.xc)*r4
        dudK3 = (X - self.xc)*r6 
        dudP1 = (r2 + 2*(X - self.xc)**2)*(1 + self.p[5]*r2 + self.p[6]*r4)
        dudP2 = 2*(X - self.xc)*(Y - self.yc)*(1 + self.p[5]*r2 + self.p[6]*r4)
        dudP3 = (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*r2 
        dudP4 = (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*r4 
        
        dvdK1 = (Y - self.yc)*r2
        dvdK2 = (Y - self.yc)*r4
        dvdK3 = (Y - self.yc)*r6
        dvdP1 = 2*(X - self.xc)*(Y - self.yc)*(1 + self.p[5]*r2 + self.p[6]*r4)
        dvdP2 = (r2 + 2*(Y - self.yc)**2)*(1 + self.p[5]*r2 + self.p[6]*r4)
        dvdP3 = (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*r2 
        dvdP4 = (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*r4 
        
        
        return np.vstack((dudK1,dudK2,dudK3,dudP1,dudP2,dudP3,dudP4)).T, np.vstack((dvdK1,dvdK2,dvdK3,dvdP1,dvdP2,dvdP3,dvdP4)).T 




class ParametricCameraBisBis:
    """
    Brown-Conrady parametric distortion model 
    """ 
    def __init__(self, p, xc, yc):
        self.p  = p 
        self.xc = xc 
        self.yc = yc 

    def P(self,X,Y):
        # rho2 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**1 
        # rho4 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**2 
        # rho6 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**3 
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2  

        u = X + self.p[7]*(X - self.xc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 ) + self.p[8]*(self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*(1 + self.p[5]*r2 + self.p[6]*r4)
        v = Y + self.p[9]*(Y - self.yc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 ) + self.p[10]*(2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*(1 + self.p[5]*r2 + self.p[6]*r4)   
        
        return u,v 
    
    def dPdX(self,X,Y):
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2 
        
        K1,K2,K3,P1,P2,P3,P4 = self.p[:7]
 
        
        dudx = 1 + self.p[7]* ( K1*r2 + K2*r4 + K3*r6 + (X - self.xc)**2*(2*K1 + 4*K2*r2 + 6*K3*r4) ) +\
               self.p[8]*( (6*P1*(X - self.xc) + 2*P2*(Y - self.yc))*(1 + P3*r2 + P4*r4) +\
               (P1*(r2 + 2*(X - self.xc)**2) + 2*P2*(X - self.xc)*(Y - self.yc))*(2*P3*(X - self.xc) + 4*P4*(X - self.xc)*r2)    )
        
        dudy = self.p[7]*(X - self.xc)*(Y - self.yc)*(2*K1 + 4*K2*r2 + 6*K3*r4) + self.p[8]*(2*P1*(Y - self.yc) + (2*P2*(X - self.xc))*(1 + P3*r2 + P4*r4) +\
               (P1*(r2 + 2*(X - self.xc)**2) + 2*P2*(X - self.xc)*(Y - self.yc))*(2*P3*(Y - self.yc) + 4*P4*(Y - self.yc)*r2))
        
        dvdx =  self.p[9]*(X - self.xc)*(Y - self.yc)*(2*K1 + 4*K2*r2 + 6*K3*r4) + self.p[10]*(2*P1*(Y - self.yc) + (2*P2*(X - self.xc))*(1 + P3*r2 + P4*r4) +\
               (P1*(r2 + 2*(X - self.xc)**2) + 2*P2*(X - self.xc)*(Y - self.yc))*(2*P3*(Y - self.yc) + 4*P4*(Y - self.yc)*r2))
        
        dvdy =  1 + self.p[9]*( K1*r2 + K2*r4 + K3*r6 + (Y - self.yc)**2*(2*K1 + 4*K2*r2 + 6*K3*r4) )   +\
                self.p[10]*( (6*P1*(Y - self.yc) + 2*P2*(X - self.xc))*(1 + P3*r2 + P4*r4) +\
                (P1*(r2 + 2*(Y - self.yc)**2) + 2*P2*(Y - self.yc)*(X - self.xc))*(2*P3*(Y - self.yc) + 4*P4*(Y - self.yc)*r2)   ) 

        
        return  dudx, dudy, dvdx, dvdy 
    
    def dPdp(self,X,Y):
        r2 = (X-self.xc)**2 +(Y-self.yc)**2
        r4 = r2*r2 
        r6 = r4*r2 
        
        dudK1 = (X - self.xc)*r2 * self.p[7]
        dudK2 = (X - self.xc)*r4 * self.p[7]
        dudK3 = (X - self.xc)*r6 * self.p[7]
        dudP1 = ( (r2 + 2*(X - self.xc)**2)*(1 + self.p[5]*r2 + self.p[6]*r4) ) * self.p[8]
        dudP2 = ( 2*(X - self.xc)*(Y - self.yc)*(1 + self.p[5]*r2 + self.p[6]*r4) )*self.p[8]
        dudP3 = (  (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*r2 ) * self.p[8] 
        dudP4 = ( (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*r4 ) * self.p[8]
        dudP5 = (X - self.xc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 )
        dudP6 = (self.p[3]*(r2 + 2*(X - self.xc)**2) + 2*self.p[4]*(X - self.xc)*(Y - self.yc))*(1 + self.p[5]*r2 + self.p[6]*r4)
        dudP7 = np.zeros(len(X)) 
        dudP8 = np.zeros(len(X))
        
        dvdK1 = (Y - self.yc)*r2 * self.p[9]
        dvdK2 = (Y - self.yc)*r4 * self.p[9]
        dvdK3 = (Y - self.yc)*r6 * self.p[9]
        dvdP1 = ( 2*(X - self.xc)*(Y - self.yc)*(1 + self.p[5]*r2 + self.p[6]*r4) ) * self.p[10]
        dvdP2 = ( (r2 + 2*(Y - self.yc)**2)*(1 + self.p[5]*r2 + self.p[6]*r4) ) * self.p[10]
        dvdP3 = ( (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*r2 ) * self.p[10] 
        dvdP4 = ( (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*r4 ) * self.p[10] 
        dvdP5 = np.zeros(len(X))
        dvdP6 = np.zeros(len(X))
        dvdP7 = (Y - self.yc)*(self.p[0]*r2 + self.p[1]*r4 + self.p[2]*r6 ) 
        dvdP8 = (2*self.p[3]*(X - self.xc)*(Y - self.yc) + self.p[4]*(r2 + 2*(Y - self.yc)**2))*(1 + self.p[5]*r2 + self.p[6]*r4) 
        
        
        return np.vstack((dudK1,dudK2,dudK3,dudP1,dudP2,dudP3,dudP4,dudP5,dudP6,dudP7,dudP8)).T \
               ,np.vstack((dvdK1,dvdK2,dvdK3,dvdP1,dvdP2,dvdP3,dvdP4,dvdP5,dvdP6,dvdP7,dvdP8)).T 
    
    
 

        
    

    
class ParametricCamera:
    """
    Brown-Conrady parametric distortion model 
    """    
    def __init__(self, p, xc, yc):
        self.p  = p 
        self.xc = xc 
        self.yc = yc 
        
    def P(self,X,Y):
        # rho2 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**1 
        # rho4 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**2 
        # rho6 = ( (X-self.xc)**2 +(Y-self.yc)**2 )**3 
        rho2 = (X-self.xc)**2 +(Y-self.yc)**2
        rho4 = rho2*rho2 
        rho6 = rho4*rho2 
        
        r1,r2,r3,d1,d2 = self.p
        
        u = X + (X - self.xc)* ( r1*rho2 + r2*rho4 + r3*rho6 ) + 2*d1*(X-self.xc)*(Y-self.yc) + d2*( rho2 + 2*(X-self.xc)**2 )
        v = Y + (Y - self.yc)* ( r1*rho2 + r2*rho4 + r3*rho6 ) + 2*d2*(X-self.xc)*(Y-self.yc) + d1*( rho2 + 2*(Y-self.yc)**2 )
        return u,v 
 
    def dPdX(self,X,Y):

        rho2 = (X - self.xc)**2 +(Y - self.yc)**2
        rho4 = rho2*rho2 
        rho6 = rho4*rho2 
        
        r1,r2,r3,d1,d2 = self.p
        
        dudx =   1 + r1*rho2 + r2*rho4 + r3*rho6 + (X - self.xc)**2*(2*r1 + 4*r2*rho2 + 6*r3*rho4) + 6*d2*(X - self.xc) + 2*d1*(Y - self.yc)
        dudy =   (X - self.xc)*(Y - self.yc)*(2*r1 + 4*r2*rho2 + 6*r3*rho4) + 2*d2*(Y - self.yc) + 2*d1*(X - self.xc) 
        dvdx =   dudy 
        dvdy =   1 + r1*rho2 + r2*rho4 + r3*rho6 + (Y - self.yc)**2 * (2*r1 + 4*r2*rho2 + 6*r3*rho4) + 2*d2*(X - self.xc) + 6*d1*(Y - self.yc) 
        return  dudx, dudy, dvdx, dvdy
    
 
    def dPdp(self,X,Y):
        
        rho2 = (X-self.xc)**2 +(Y-self.yc)**2
        rho4 = rho2*rho2 
        rho6 = rho4*rho2 
        
        r1,r2,r3,d1,d2 = self.p
        
        dudr1 =  (X-self.xc)*rho2 
        dudr2 =  (X-self.xc)*rho4
        dudr3 =  (X-self.xc)*rho6  
        
        dvdr1 =  (Y-self.yc)*rho2 
        dvdr2 =  (Y-self.yc)*rho4 
        dvdr3 =  (Y-self.yc)*rho6 
        
        dudd2 =  rho2 + 2*(X-self.xc)**2 
        dudd1 =  2*(X-self.xc)*(Y-self.yc) 

        dvdd2 =  dudd1  
        dvdd1 =  rho2 + 2*(Y-self.yc)**2  
        
        return np.vstack((dudr1,dudr2,dudr3,dudd1,dudd2)).T, np.vstack((dvdr1,dvdr2,dvdr3,dvdd1,dvdd2)).T 




    
def GetCamFromData(X,Y,Px,Py,xc,yc,mx,my,CameraFunc):
    """
    Returns the Camera model that fits the 
    best the Field (Px,Py) defined on the data points 
    (X,Y)
    """
    
    def residual(p,x,f):
        cam = CameraFunc(p, xc, yc, mx, my)
        Mx,My = cam.P(x[:,0],x[:,1])
        model = np.r_[Mx,My]
        return f-model 

    def Jac_residual(p,x,f):
        cam = CameraFunc(p, xc, yc, mx, my)
        dMx,dMy =  cam.dPdp(x[:,0],x[:,1])
        return -np.vstack((dMx,dMy)) 
    
    
    p0 = np.zeros(len(mx)+len(my)) # Initial start for parameters 
    result = sp.optimize.least_squares(residual,
                                       p0,  
                                       args=(np.c_[X,Y], np.r_[Px, Py] ),
                                       jac= Jac_residual )
 
    
    
    p = result['x']
    cam = CameraFunc(p, xc, yc, mx, my)
    return cam 


def GetCamFromOneImage(images,rois,tx,ty,cam,Niter,tol):
    """
    images: center, right, left, up, down
    rois: overlapping regions 
    proj: used projector 
    tx: image translations in x  (size 4)  # Attention tx<0
    ty: image translations in y  (size 4)  # Attention ty<0
    """
    
    """
    Setting the data points for the four regions 
    """
    xtot = []
    ytot = [] 
    # plt.figure() 
    # plt.imshow(images[0].pix, cmap='gray')
    for i in range(4):
        x1d = np.arange(rois[i][0,0],rois[i][1,0])
        y1d = np.arange(rois[i][0,1],rois[i][1,1])
        X,Y = np.meshgrid(x1d,y1d,indexing='ij') 
        xtot.append(X.ravel())
        ytot.append(Y.ravel())
    #     plt.plot(xtot[i],ytot[i],'.',markersize=2)
    # raise ValueError('Stop')
    p = cam.p 
    m = len(cam.p)
    for i in range(4):
        p = np.r_[p,tx[i],ty[i]]
    f1 = images[0] # Reference image is the central image
    # GN iterations 
    for ik in range(Niter):
        H   = np.zeros((len(p),len(p)))
        b   = np.zeros(len(p))
        res_tot = [None]*4 
        # Loop over the overlapping regions 
        for i in range(4):
            f2 = images[i+1]
            x = xtot[i]
            y = ytot[i]
            pgu1, pgv1   = cam.P(x,y) 
            pgu2, pgv2   = cam.P(x+p[m+2*i],y+p[m+2*i+1])  
            f1p            = f1.Interp(pgu1, pgv1)
            df1dx, df1dy   = f1.InterpGrad(pgu1,pgv1)  
            f2p            = f2.Interp(pgu2, pgv2)
            df2dx, df2dy   = f2.InterpGrad(pgu2,pgv2)   
            # Jacobian of projector with respect to the distortion parameters 
            Jpu1, Jpv1       = cam.dPdp(x,y)
            
            Jxu1      = np.zeros((len(x),8))
            Jyv1      = np.zeros((len(x),8))
            Jpu1 = np.c_[Jpu1,Jxu1]
            Jpv1 = np.c_[Jpv1,Jyv1]
            
            Jxu2          = np.zeros((len(x),8))
            Jyv2          = np.zeros((len(x),8))
            dudx, dudy, dvdx, dvdy = cam.dPdX(x+p[m+2*i],y+p[m+2*i+1])  
            Jxu2[:,2*i]   = dudx 
            Jyv2[:,2*i+1] = dvdy 
            
            Jpu2, Jpv2       = cam.dPdp(x+p[m+2*i],y+p[m+2*i+1])
            Jpu2 = np.c_[Jpu2, Jxu2 ]
            Jpv2 = np.c_[Jpv2, Jyv2 ]            
            
            
            Jp1  = np.concatenate((Jpu1,Jpv1))
            Jp2  = np.concatenate((Jpu2,Jpv2))
            n = len(x)
            df1  =  sp.sparse.dia_matrix((np.vstack((df1dx, df1dy)), np.array([0,-n])), shape=(2*n,n))
            df2  =  sp.sparse.dia_matrix((np.vstack((df2dx, df2dy)), np.array([0,-n])), shape=(2*n,n))      
            M    =  df1.T.dot(Jp1) - df2.T.dot(Jp2)  
            H    += M.T.dot(M) 
            res  = f1p - f2p 
            res_tot[i] = res 
            b    -= M.T.dot(res) 
        dp = np.linalg.solve(H,b)
        p  += dp 
        cam.p = p[:m] 
        err = np.linalg.norm(dp)/np.linalg.norm(p)
        print("Iter # %2d | s1=%2.2f , s2=%2.2f, s3=%2.2f, s4=%2.2f | dp/p=%1.2e" 
              % (ik + 1, np.std(res_tot[0]),np.std(res_tot[1]),np.std(res_tot[2]),np.std(res_tot[3]), err))
        if err < tol:
            break
    return cam
    
def PlotRoi(roi,color):
    plt.plot([roi[0,0],roi[0,0]],[roi[0,1],roi[1,1]], color=color)
    plt.plot([roi[1,0],roi[1,0]],[roi[0,1],roi[1,1]], color=color)
    plt.plot([roi[0,0],roi[1,0]],[roi[0,1],roi[0,1]], color=color)
    plt.plot([roi[0,0],roi[1,0]],[roi[1,1],roi[1,1]], color=color)

def ProgressBar(percent):
    width = 40 
    left = width * percent // 100
    right = width - left
    
    tags = "█" * int(np.round(left))
    spaces = " " * int(np.round(right))
    percents = f"{percent:.0f}%"
    
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)
    

def ReadFijiLog(file):
    """ Reads the Fiji modified log 
    in order to get the correlation score on 
    each overlapping region 
    """
    Rfile = open(file)  
    line = Rfile.readline()
    k = 1 
    # print(k,line)
    R = [] 
    while len(line)!=0:
        # Extract the correlation coefficient 
        s1 = '='
        s2 = '('
        # print(k,line)
        i1 = line.index(s1)
        i2 = line.index(s2,i1+1) 
        Rs = line[i1+1:i2-1]
        R.append(float(Rs))
        k+=1
        line = Rfile.readline()
    return np.array(R)
    
 
    
 
def MeshFromROI(roi, dx, typel=3):
    import pyxel 
    cam = pyxel.Camera([1,0,0,np.pi/2])
    m = pyxel.StructuredMesh(roi, dx, typel=typel)
    return m, cam

 



 






    

               
    
    
