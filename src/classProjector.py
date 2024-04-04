import numpy as np 
import matplotlib.pyplot as plt 

class Projector:
    """
    Polynomial camera model 
    constant, x, y, x*y, x**2, y**2, x**2*y, x*y**2, x**3, y**3 
    """
    def __init__(self, p, xc, yc, mx, my, lx, ly):
        self.p  = p 
        self.mx = mx 
        self.my = my 
        self.xc = xc 
        self.yc = yc 
        self.lx = lx 
        self.ly = ly 
    
    def ShowValues(self):
        n = len(self.mx)
        px = self.p[:n]
        py = self.p[n:]
        modes = ['c', 'x', 'y', 'x*y', 'x**2', 'y**2', 'x**2*y', 'x*y**2', 'x**3', 'y**3' ]  
        print('X:  ', end= ''); [print(modes[self.mx[i]], end='  ') for i in range(len(self.mx))] 
        print("\n",px)
        print('Y:  ', end= ''); [print(modes[self.my[i]], end='  ') for i in range(len(self.my))] 
        print("\n",py)
        
    def P(self,X,Y): 
 
        x = (X - self.xc) / self.lx  
        y = (Y - self.yc) / self.ly  
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
        
        x = (X - self.xc) / self.lx 
        y = (Y - self.yc) / self.ly 
        dAdX = np.vstack((zero,one,zero,y,2*x,zero,2*x*y,y**2,3*x**2,zero)).T  / self.lx 
        dAdY = np.vstack((zero,zero,one,x,zero,2*y,x**2,2*x*y,zero,3*y**2)).T  / self.ly 
        
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
        x = (X - self.xc) / self.lx 
        y = (Y - self.yc) / self.ly 
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
 