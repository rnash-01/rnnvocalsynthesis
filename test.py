import numpy as np
from dataconversion import *
import scipy
import matplotlib.pyplot as plt
framerate = 44100
start = 0
seconds = 2
end = framerate * seconds
step = 1/framerate
def process():
    freq1 = 1
    freq2 = 2

    x = np.arange(start, seconds, step)
    y = 0.5*np.sin(x * freq1 * 2 * np.pi) + 0.5*np.sin(x * freq2 * 2 * np.pi)


    yf = np.absolute(scipy.fft.rfft(y))
    xf = scipy.fft.rfftfreq(end, step)

    return yf

def deprocess(data):
    samp = scipy.fft.irfft(data)
    x = np.arange(start, seconds, step)
    plt.plot(x, samp)
    plt.show()

frequencies = process()
deprocess(frequencies)
