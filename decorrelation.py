import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import pi
import scipy.signal as sgn
import time
# import os
# from PIL import Image

from myLib import *

import decorr_lib as dcr
import BW_PSF_lib as BW

#%% image simulation

t1 = time.time()

wl = 0.488
k = 2*pi/wl
n = 1.33
Z = 0
px = 0.05 # pixel size in um
NA = 0.8

N = 256
M = 256

c = [N//2, M//2]

x = np.arange(N)
y = np.arange(M)

X, Y = np.meshgrid(x,y)

R = np.sqrt( (X-c[0])**2 + (Y-c[1])**2 )*px

psf = BW.PSF(Z, R, NA, wl, n)
density = 1000
P = rnd.randint(low=0, high=density + 1, size=(N,M)) // density

Z = 1e3*sgn.fftconvolve(P, psf,'same')

bkg = 100
noise = 0
img = np.uint16( rnd.poisson(lam = Z + bkg) + rnd.normal(0, 20, np.shape(Z)) )

plt.figure(1)
plt.imshow(img)

Theo_res = 0.61*wl*n/NA
print('Theoretical Resolution =', Theo_res, 'um')

elapsed = time.time() - t1
print('Image simulation time = ', elapsed, 's')

#%% Resolution calculation

t2 = time.time()

N_points = 200
N_curves = 40
Resolution, r, A, R, D2 = dcr.Decorrelation(img, px, N_points, N_curves)

elapsed = time.time() - t2
print('Resolution calculation time = ', elapsed, 's')

plt.figure(2)
for m in np.arange(len(D2)):
    plt.plot(R,D2[m])
    plt.plot(r[m],A[m],'kx')
    
print('Measured Resolution =', Resolution, 'um')

plt.xlabel('Mask Radius')
plt.ylabel('Cross-Correlation')