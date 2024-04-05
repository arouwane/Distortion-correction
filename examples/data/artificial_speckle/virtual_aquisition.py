import PIL 
import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.interpolate as spi 
dir = os.path.dirname(os.path.abspath(__file__))
for i in range(3):
    dir = os.path.dirname(dir)
sys.path.append(dir+'/src')
import corrector   
path = os.path.abspath(os.path.dirname(sys.argv[0]))

file_name = 'speckle_image'
im1 = PIL.Image.open(path + '/' + file_name +'_ref.tif')
im1 = np.asarray(im1)    
im1Interp = spi.RectBivariateSpline(np.arange(0, im1.shape[0]) , np.arange(0, im1.shape[1]), im1) 

# We set a distortion field 
mx = [3, 4, 5,  7, 8]
my = [3, 4, 5,  6, 9]
d0 = 4 ; d1 = 6 ; d2 = 8 ; 
p = np.array([2*d2, 3*d1, d1, d0, d0, 2*d1, d2, 3*d2, d0, d0])
s = 1024 # Size of a sub-image 
xc = s/2 
yc = s/2 
D = corrector.Projector(p, xc, yc, mx, my, s, s)  # The original correction field we are looking for 

plt.figure()
D.Plot(size=(s,s))
plt.show() 



# Getting the distorted points 
x = np.arange(s)
y = np.arange(s)
X,Y = np.meshgrid(x,y,indexing='ij')
px, py = D.Pinv( X.ravel() , Y.ravel()  )
# px, py = camt.P(X.ravel(), Y.ravel())
  
ni = 3 
nj = 3   
ol = 200 
nx = ni 
ny = ni 
i0 = 512 
j0 = 512  
images0 = [None] * nx * ny  
for i in range(1, ni**2 + 1):
    iy = np.ceil(i/ny)  
    ix = iy%2*(i-(iy-1)*ny) + (iy-1)%2*(ny+1+(iy-1)*ny-i)
    ty = (ix-1) * ( s - ol )   
    tx = (iy-1) * ( s - ol ) 
    print("Image %d, tx = %f, ty = %f " %(i,tx,ty))
    # Extracting the sub-images 
    images0[i-1]     = corrector.Image(None)
    imin = int(tx) ;  
    jmin = int(ty) ;  
    images0[i-1].pix = im1Interp.ev(px + imin + i0, py + jmin + j0 ).reshape((s,s)) 
    PILimg = PIL.Image.fromarray(np.round(images0[i-1].pix).astype("uint8"))
    PILimg.save(path + '/' + file_name + '_A%03d.tif'%i) 