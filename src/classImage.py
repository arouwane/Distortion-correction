from classBilinearInterpolator import BiLinearRegularGridInterpolator 
import scipy.interpolate as spi
import cv2 as cv
from   scipy.ndimage import gaussian_filter
import numpy as np 
import PIL.Image as image
import matplotlib.pyplot as plt 
import os  

class Image:
    def __init__(self, fname):
        """Contructor"""
        self.fname = fname
        self.tx = None  # x translation of the image (vertical position)
        self.ty = None  # y translation of the image (horizontal position)
        self.ix = None  # x Index of the image 
        self.iy = None  # y Index of the image 
        self.txr = None # Relative shift to upper image  
        self.tyr = None # Relative shift to left image  
        self.pix = None 
        
    
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
        if method == 'bilinear':
            self.interp = BiLinearRegularGridInterpolator()
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
        if self.interpMethod == 'bilinear':
            return self.interp.evaluate(self.pix, np.vstack((x,y))) 
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
        if self.interpMethod == 'bilinear':
            return self.interp.grad(self.pix, np.vstack((x,y))) 
        
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
 