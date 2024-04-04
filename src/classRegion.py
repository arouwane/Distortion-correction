import matplotlib.pyplot as plt 
import numpy as np 
import skimage
import scipy as sp 


class Region:
    """
    Overlapping region 
    It is defined between two images 
    """
    def __init__(self,im,index_im,band_type):
        self.im0  = im[0]  #Left or bottom image 
        self.im1  = im[1]  # Right or top image 
        self.index_im0 = index_im[0] # Index of the reference image in the list of the images of the grid 
        self.index_im1 = index_im[1] # Index of the deformed image in the list of the images of the grid 
        
        self.type = band_type # Horizontal 'h' or vertical 'v' 

        # Boundaries of the region 
        self.xmin = None 
        self.xmax = None 
        self.ymin = None 
        self.ymax = None 
        
        self.bounds  = None  
        # Lengths of the region 
        self.hx      = None 
        self.hy      = None   
        
        
        # [[xmin,ymin],[xmax,ymax]]
        # xmin = bounds[0,0]
        # xmax = bounds[1,0]
        # ymin = bounds[0,1]
        # ymax = bounds[1,1]
    
    def SetBounds(self, epsilon):
        """
        If the positions (i.e. the translations) of the images are set 
        then this function defines the overlapping regions 
        """
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
        """
        Discretizes the region by sampling 
        pixel points 
        """
        x1d = np.arange( self.xmin, self.xmax )[::s]
        y1d = np.arange( self.ymin, self.ymax )[::s]
        X,Y = np.meshgrid(x1d,y1d,indexing='ij') 
        self.pgx = X.ravel() 
        self.pgy = Y.ravel() 
   
    def PlotResidual(self,res):
        """
        Plots the residual field 
        """
        shape = (self.xmax-self.xmin, self.ymax-self.ymin)
        
        plt.figure()
        plt.imshow(res.reshape(shape),cmap='RdBu')
 
    def PlotReferenceImage(self,cam=None):
        """
        Plots the left (or bottom) image 
        """
        shape = (self.xmax-self.xmin,self.ymax-self.ymin)
        if cam is None:
            pgu,pgv = self.pgx - self.im0.tx, self.pgy - self.im0.ty
        else:
            pgu, pgv   = cam.P(self.pgx-self.im0.tx,self.pgy-self.im0.ty) 
       
        plt.figure()
        plt.imshow(self.im0.Interp(pgu,pgv).reshape(shape),cmap='gray')
      
 
    def PlotDeformedImage(self,cam=None):
        """
        Plots the right (or top) image 
        """
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
        """
        Assembly of the local operators 
        for the  Gauss-Newton linear system 
        """
        
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