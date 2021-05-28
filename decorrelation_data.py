import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import pi
import scipy.signal as sgn
from PIL import Image
import time

import decorr_lib as dcr
import BW_PSF_lib as BW

#%% Select image and pixel size

import tkinter as tk
from tkinter import filedialog

app1 = tk.Tk()
app1.withdraw()
app1.attributes("-topmost", True)

file = filedialog.askopenfilename(parent=app1, title="Please select a file:")

del app1

app2 = tk.Tk()
app2.geometry('300x40')
app2.attributes("-topmost", True)
app2.focus_force()

def get_input():
    global px
    px = float( e.get() )
    app2.quit()
    app2.destroy()
    
tk.Label(app2, text="Pixel size in um:  ").grid(row=0)
e = tk.Entry(app2)
e.insert(0,  0.100)
e.grid(row=0, column=1)

tk.Button(app2, text='Ok', command=get_input).grid(row=0, column=2, sticky=tk.W, pady=4)
                                    
app2.mainloop()

del e, app2

#%% Show image

img = Image.open(file)
img = np.asarray(img)

plt.figure(1)
plt.imshow(img)

#%% Resolution calculation

N_points = 200
N_curves = 100

Resolution, r, A, R, D2 = dcr.Decorrelation(img, px, N_points, N_curves)

#%% Plot results

plt.figure(2)
for m in np.arange(len(D2)):
    plt.plot(R,D2[m])
    plt.plot(r[m],A[m],'kx')
    
print('Measured Resolution =', Resolution, 'um')

plt.xlabel('Mask Radius')
plt.ylabel('Cross-Correlation')