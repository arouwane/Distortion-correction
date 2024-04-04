from classProjector import Projector
import skimage                   
from multiprocessing import Pool 
import concurrent.futures
import matplotlib.pyplot as plt 
import numpy as np 
import skimage
import pathlib
import datetime 


def SetGlobalHorizontalShift(im0, im1, l):
    """
    Compares two consecutive images and sets 
    the position of the right image 
    """
 
    I0c = im0.pix[:,-l:]
    I1c = im1.pix[:, :l]
    shift,_,_ = skimage.registration.phase_cross_correlation( I0c, I1c, upsample_factor=100 ) 
    ti = shift[0] 
    tj = im0.pix.shape[1] - I0c.shape[1]  + shift[1] 
     
    im1.SetCoordinates(  im0.tx + ti  ,  im0.ty + tj )
     
    
def SetGlobalVerticalShift(im0, im1, l):
    """
    Compares two consecutives images and sets 
    the position of the bottom image 
    """
    
    I0c  =  im0.pix[-l:,:]   
    I1c  =  im1.pix[:l, :]   
    
    shift,_,_ = skimage.registration.phase_cross_correlation( I0c, I1c, upsample_factor=100 ) 
    ti = im1.pix.shape[0] - I1c.shape[0] + shift[0]
    tj = shift[1] 
    
    im1.SetCoordinates( im0.tx + ti,  im0.ty + tj )
    
def ProgressBar(percent):
    width = 40 
    left = width * percent // 100
    right = width - left
    
    tags = "█" * int(np.round(left))
    spaces = " " * int(np.round(right))
    percents = f"{percent:.0f}%"
    
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)
    
    

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
        ATTENTION: ONLY ONE CONSIDERED ORDER (SNAKE BY ROWS - RIGHT & DOWN)
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
        """ 
        Builds interpolation scheme for all the images 
        """
        for im in self.images:
            im.BuildInterp(method)

    def GaussianFilter(self, sigma):
        """
        Filters all the images of the grid 
        """
        for im in self.images:
            im.GaussianFilter(sigma) 
            
    def LoadImages(self):
        """ 
        Loads the images of the mosaic
        """
        for im in self.images:
            im.Load() 
            
    def Connectivity(self,conn):
        """
        Sets the vector that selects the modes and translation in the 
        toral Degree of Freedom vector
        """
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
 
    
    def PlotImageBoxes(self, origin=(0,0),eps=(0,0), color='green'):
        """
        Plots the overlapping regions 
        """
        # A = np.zeros( ( int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0], 
        #                 int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1]  ) )
        # plt.imshow(A)
        for im in self.images:
            im.PlotBox(origin,color)  
    
    def StitchImages(self, cam=None, origin=(0,0), eps=(0,0), fusion_mode='average'):
        """
        Stitch the grid into one image 
        """
        
        camc = None 
        if cam is None: 
            camc = Projector( np.array([0,0]), 0, 0, [0], [0], 1, 1 ) # Identity projector  
        else: 
            camc = cam 
            
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
                u,v      = camc.P( x[index1]-origin[0]-im.tx, y[index1]-origin[1]-im.ty )
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
                    u, v        = camc.P( x[index0]-origin[0]-r.im0.tx, y[index0]-origin[1]-r.im0.ty )
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
                    u, v   = camc.P( x[index0]-origin[0]-r.im1.tx, y[index0]-origin[1]-r.im1.ty )
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
                    u, v       = camc.P( x[index0]-origin[0]-r.im0.tx, y[index0]-origin[1]-r.im0.ty )
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
                    u, v   = camc.P( x[index0]-origin[0]-r.im1.tx, y[index0]-origin[1]-r.im1.ty )
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
        """
        Returns the Gauss-Newton side members 
        """
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
        """
        Plot the difference between neighboring images 
        on their overlapping regions in order 
        to reveal the effect of the distortion 
        """
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
            
    
    def SetTranslations(self, alphax=0.8, alphay = 0.8 ):
        """
        Finds the translation between the images 
        SHould be improved 
        Gives less accurate results than Fiji 
        """
        # alphax is parameter of reducing the 
        # size of the overlapping regions in order 
        
        lx = int( np.floor(alphax * self.sx * self.ox / 100) ) 
        ly = int( np.floor(alphay * self.sy * self.oy / 100) )  
  
        
        # Loop over images and 
        # Set global translations 
        # For the image stitching 
        for i in range(self.nx):
            # Loop over a row 
            # If even row, indices increase 
            print("Row "+str(i))
            
            if i != 0 : 
                if i%2 == 0: 
                    j1 =  self.ny -1  
                    a1 =  j1  + (i-1) * self.ny 
                    j2 =  0 
                    a2 =  j2  + i * self.ny  
                    print("**Comparing the images %d %d " %(a1,a2)) 
                    SetGlobalVerticalShift(self.images[a1], self.images[a2], lx)
                else: 
                    j1 =  0  
                    a1 =  j1  + (i-1) * self.ny 
                    j2 =  self.ny - 1   
                    a2 =  j2  + i * self.ny 
                    print("**Comparing the images %d %d " %(a1,a2)) 
                    SetGlobalVerticalShift(self.images[a1], self.images[a2], lx)
                
            if i%2 == 0:
                for j in range(self.ny - 1):
                    a =  j + i * self.ny 
                    a1 = a
                    a2 = a + 1
                    print("**Comparing the images %d %d " %(a1,a2)) 
                    SetGlobalHorizontalShift(self.images[a1], self.images[a2], ly )
            else: 
                for j in range(self.ny - 1, 0, -1):
                    a =  j  + i * self.ny 
                    a1 = a
                    a2 = a - 1 
                    print("**Comparing the images %d %d " %(a1,a2)) 
                    SetGlobalHorizontalShift(self.images[a1], self.images[a2], ly )
                    

