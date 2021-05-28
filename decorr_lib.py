import numpy as np
from scipy import pi
import scipy.signal as sgn
import scipy.fft as ft
from statsmodels.nonparametric.smoothers_lowess import lowess

def hann2d(*args):
    if len(args)>1 :
        N, M = args
    else:
        N, = args
        M=N

    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*pi*X)/(N-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*pi*Y)/(M-1) ) )
    return W

def smooth(x,y):
    filtered = lowess(y, x, is_sorted=True, frac=0.05, it=0)
    return filtered[:,1]

def Normalize(img):
    Nx, Ny = np.shape(img)
    img = img - np.mean(img)
    img = img * hann2d(Nx,Ny)
    
    ft_1 = ft.fftshift( ft.fft2( img ) )
    ft_2 = ft_1/np.abs(ft_1)
    
    # In = ft.ifft2( ft_img )
    
    return np.real( ft_1 ), np.real( ft_2 )

def Mask(Nx, Ny, R_max):
    c = [Nx//2, Ny//2]

    x = np.arange(Nx)
    y = np.arange(Ny)
    
    X, Y = np.meshgrid(x,y)
    
    R = np.sqrt( (X-c[0])**2 + (Y-c[1])**2 ) / np.min(c)
    
    mask = (R < R_max).astype(np.float64)
    
    return mask

def Decorr_1(I, I_n, num):
    Nx, Ny = np.shape(I)
    D  = np.zeros(num)
    R = np.linspace(1/num,1,num)
    
    for j, r in enumerate( R ):
        M = Mask(Nx, Ny, r)
        num = np.mean( np.real( I*M*np.conj(I_n) ) )
        den = np.sqrt( np.mean( np.abs(M*I_n)**2 ) * np.mean( np.abs(I)**2 ) )
        D[j] = num/den
        # plt.figure()
        # plt.imshow(M)
    return R, D

def High_pass(img, t, s):
    Nx, Ny = np.shape(img)
    cx = np.int ( ( Nx + np.mod(Nx,2) ) / 2)
    cy = np.int ( ( Ny + np.mod(Ny,2) ) / 2)

    x = np.arange(Nx)
    y = np.arange(Ny)
    
    X, Y = np.meshgrid(x,y)
    R = np.sqrt( (X-cx)**2 + (Y-cy)**2 )
    
    S = s*np.min([cx,cy])
    T = t*np.min([cx,cy])
    sigmoid = 1 / (1 + np.exp( -(R-T)/S ))
    
    img_filt = img * sigmoid
    
    return img_filt

def Decorrelation(img, px, N_lp = 500, N_hp = 200):
    I, I_n = Normalize( img )
    
    r = []
    A = []
    D2 = []
    
    T = np.linspace(0, 1, N_hp)
    
    for t in T:
        I_t = High_pass(I, t, 0.02)
        R, D = Decorr_1(I_t, I_n, N_lp)
        D = smooth(R, D)
        p, _ = sgn.find_peaks(D, prominence = 0.001*np.max(D), width = 0.001*len(D))
        try:
            idx = np.argmax(D[p])
            idx = p[idx]
            r.append( R[idx] )
            A.append( D[idx] )
            D2.append(D)
        except ValueError:
            break
    
    Resolution = 2*px/np.max(r)
    r = np.asarray(r)
    A = np.asarray(A)
    
    return Resolution, r, A, R, D2