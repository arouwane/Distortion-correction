from classRegion import Region 
from classImage import Image
from classGrid import Grid
from classProjector import Projector
import numpy as np  
 

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
    


def CreateGrid(input_param, images=None):
    """
    Creates the grid object representing the image mosaic 
    Input: - Directorty of images 
           - input_param directory  
    """
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
    nx                   = input_param['nx']
    ny                   = input_param['ny']
    
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
            images[i] = Image(dire+file_name+"%03d" % (a)+'_'+im_type + extension ) 
            images[i].Load() 
            ix = ix_list[i]
            iy = iy_list[i]
            tx = (ix-1) * np.floor( sy - sy*oy/100)
            ty = (iy-1) * np.floor( sx - sx*ox/100)
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
            
            regions[k] = Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'v')
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
            
            regions[k] = Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'h')
            k+=1
    
    grid = Grid((nx,ny),(ox,oy),(sx,sy),images,regions)
    return grid 
            
 
def DistortionAdjustment(input_param, cam, images, epsilon=0 ): 
    """
    Identifies the distortion function 
    Takes as input 
    An initial projector cam 
    The set of images 
    epsilon is a croping parameter in order to reducing the overlapping regions 
    """
    # etting the registration parameters 
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
    if d0 is None: 
        d0 = np.zeros(len(mx)+len(my))

    dire                 = input_param['dire']
    file_name            = input_param['file_name']
    
    nx                   = input_param['nx']
    ny                   = input_param['ny']
 
 
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
            images[i] = Image(dire+file_name+"%03d" % (a)+'_'+im_type + extension ) 
            images[i].Load() 
            ix = ix_list[i]
            iy = iy_list[i]
            tx = (ix-1) * np.floor( sy - sy*oy/100)
            ty = (iy-1) * np.floor( sx - sx*ox/100)
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
            
            regions[k] = Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'v')
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
            
            regions[k] = Region((images[a0-1], images[a1-1]),(a0-1,a1-1),'h')
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
        
        cam  = Projector(d0, sx/2, sy/2, mx, my, sx, sy)
        
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
        

    
    grid = Grid((nx,ny),(ox,oy),(sx,sy),im_list,r_list)
    
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
            r.SetBounds(epsilon)
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
        print('Mean residual std on regions: %2.2f' % np.mean(Res[:,ik]) )
        print('Std residual std on regions: %2.2f' % np.std(Res[:,ik]))
        print('Maximal residual std on regions: %2.2f '% np.max(Res[:,ik]))
        print('Minimal residual std on regions: %2.2f '% np.min(Res[:,ik]))

        print('----------------------------------------')
        

            
        if err < tol:
            break
        
    return cam, images, grid, res_tot  




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
    