"""    @author: Muhammad Ali Qureshi   """

""" working on this portion for voice encryption"""

# Importing Modules
from matplotlib import pyplot as plt
import numpy as np
import moviepy.editor as mp
import scipy.io.wavfile as wavf

# Load Video File 
clip = mp.VideoFileClip("C:/...mp4")

# Load Extracted Audio from Video
clip.audio.write_audiofile("C:/...theaudio.wav")

# Load to Sample Audio Data
samplerate, data = wavf.read('C:/...theaudio.wav')

# Extracting and Sampling Voice Data
times = np.arange(len(data))/float(samplerate)

# Slicing Voice Data
data0=data[:,0]
cut1=data0

# Lorenz Attractor for Generating Chaos
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

dt = 0.01
num_steps = len(cut1)-1

# Need One More For The Initial Values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set Initial Values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

xr=xs;yr=ys;zr=zs

# Select Time and Wave
time=times; wave=cut1

xr=np.int16(xr); yr=np.int16(yr); zr=np.int16(zr)
xr=xr*10000; yr=yr*10000; zr=zr*10000

"""Encryption Portion"""
x=np.bitwise_xor(xr,yr)
x1=np.bitwise_xor(x,zr)
encr1=np.bitwise_xor(x1,wave) 

"""Decryption Portion"""
decr1=np.bitwise_xor(encr1,x1)
encr=encr1
decr=decr1

"""Ploting Portion for Voice and Magnitude Spectrum"""
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axes[0, 0].set_title("Original Voice")
axes[0, 0].plot(time,wave, color='red')
axes[0, 1].set_title("Encrypted Voice")
axes[0, 1].plot(time,encr, color='orange')
axes[0, 2].set_title("Decrypted Voice")
axes[0, 2].plot(time,decr, color='green')
axes[1, 0].set_title("Magnitude Spectrum Original Voice")
axes[1, 0].magnitude_spectrum(wave, Fs=2, color='red')
axes[1, 1].set_title("Magnitude Spectrum Encrypted Voice ")
axes[1, 1].magnitude_spectrum(encr, Fs=2, color='orange')
axes[1, 2].set_title("Magnitude Spectrum Decrypted Voice")
axes[1, 2].magnitude_spectrum(decr, Fs=2, color='green')
fig.tight_layout()
plt.show()

"""Ploting Portion for Power Spectrum Density and Spectrogram"""
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axes[0, 0].set_title("Power Spectrum Density Original Voice")
axes[0, 0].psd(wave, Fs=2, color='red',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[0, 1].set_title("Power Spectrum Density Encrypted Voice ")
axes[0, 1].psd(encr, Fs=2, color='orange',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[0, 2].set_title("Power Spectrum Density Decrypted Voice")
axes[0, 2].psd(decr, Fs=2, color='green',pad_to=1024,scale_by_freq=True,NFFT=1024)
axes[1, 0].set_title("Spectrogram Original Voice")
axes[1, 0].specgram(wave, Fs=4)
axes[1, 1].set_title("Spectrogram Original Encrypted Voice ")
axes[1, 1].specgram(encr, Fs=4)
axes[1, 2].set_title("Spectrogram Original Decrypted Voice")
axes[1, 2].specgram(decr, Fs=4)
fig.tight_layout()
plt.show()

# Generating Output for required Wavefile
if __name__ == "__main__":
    samples = decr
    # samples = np.random.randn(44100)
    fs = 44100
    
    # Write Audio File
    out_f = 'C:/...theaudio d.mp3'
    wavf.write(out_f, fs, samples)
    
#%%
""" working on this portion for video encryption"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# Load Video File 
path = "C:/...intel.mp4"
for file in glob.glob(path):
    cap = cv2.VideoCapture(file)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:/...thevideo d.mp4',fourcc, 20.0, (w,h))
num=w*h

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            '''
            Given:
               x, y, z: a point of interest in three dimensional space
               s, r, b: parameters defining the lorenz attractor
            Returns:
               x_dot, y_dot, z_dot: values of the lorenz attractor's partial
                   derivatives at the point x, y, z
            '''
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return x_dot, y_dot, z_dot
        
        dt = 0.01
        num_steps = num
        
        # Need one more for the initial values
        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)
        
        # Set initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)
        
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
        
        xr=xs;yr=ys;zr=zs
        
        # Framing RGB
        b, g, r    = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] # For RGB image
        xr1=np.array(xr[0:num]*10000);yr1=np.array(yr[0:num]*10000);zr1=np.array(zr[0:num]*10000)
        rr=r;bb=b;gg=g
        xr2=abs(np.int16(xr1));yr2=abs(np.int16(yr1));zr2=abs(np.int16(zr1))
        
        # Reshape Image Data
        xrr=xr2.reshape(h,w)
        yrr=yr2.reshape(h,w)
        zrr=zr2.reshape(h,w)
        
        """encr portion"""
        R=np.bitwise_xor(rr,xrr)
        G=np.bitwise_xor(gg,yrr)
        B=np.bitwise_xor(bb,zrr)
        R1=R.astype(np.uint8)
        G1=G.astype(np.uint8)
        B1=B.astype(np.uint8)
        """decr portion"""
        R11=np.bitwise_xor(R,xrr)
        G11=np.bitwise_xor(G,yrr)
        B11=np.bitwise_xor(B,zrr)
        dR=R11.astype(np.uint8)
        dG=G11.astype(np.uint8)
        dB=B11.astype(np.uint8)

        """ output making portion """     
        im2 = cv2.merge((dR, dG, dB))[...,::-1]        
        im = cv2.merge((R1, G1, B1))

        # out.write(frame)
        # out.write(im)
        out.write(im2)

        # cv2.imshow('original',frame)
        # cv2.imshow('encrypted',im)
        cv2.imshow('decrypted',im2)
        if cv2.waitKey(32) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
#%%

"Merging audio and video data"
from moviepy.editor import *

# Load Audio and Video File 
videoclip = VideoFileClip("C:/...thevideo d.mp4") 
audioclip = AudioFileClip("C:/...heaudio d.mp3")

# New Video Destination
new_audioclip = CompositeAudioClip([audioclip])
videoclip.audio = new_audioclip
videoclip.write_videofile("C:/...Decrypted Video.mp4")