################################################################################

# This is the main file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated:

################################################################################

#                              # IMPORTS #                                 #
from scipy.fft import rfft, rfftfreq, irfft
import rnn, wave, numpy, math
from dataconversion import *
import matplotlib.pyplot as plt
import numpy as np
import time
################################################################################

#                          # GLOBAL CONSTANTS #                            #

TEST = True

SAMPLE_SIZE = 0  # How large a given audio sample will be. Max size.

# How large the input layer will be/how many frequency bands will be passed
# into it.
INPUT_SIZE = 0

# A few parameters for the storage of frequency data
START_FREQ = 100
STOP_FREQ = 1000
INTERVAL = 100
MAX_AMP = 1

# The name of the voice (example, for the time being):
VNAME = "test"

################################################################################
def getFname(type):
    #name = input("Please enter the name of {0}: ".format(type))

    # ** Write validation checks here **

    return name

def decodeAudio(hexstring, signed, endianness, p):
    binarr = []
    for i in range(len(hexstring)//2):
        currenthex = hexstring[(i*2):(i*2)+2]
        binarr += hexToBinary(currenthex, endianness)
    if p:
        print(binarr)
    return binaryToDenary(binarr, signed)



def encodeAudio(frequencies):
    pass

def getAudio(intype, fname):
    global MAX_AMP
    f = wave.open(fname, 'rb')

    # Get some parameters of the file
    allparams = f.getparams()
    channels = f.getnchannels()
    sampwidth = f.getsampwidth()        # Bytes per sample
    framerate = f.getframerate()        # Frames per second
    n = f.getnframes()                  # Total number of frames

    # Create some of my own parameters
    sampsize = int(framerate * 0.25)    # Frames per sample
    readlim = math.ceil(n/sampsize)     # Number of samples we'll read

    # Read in all data into samples
    samples = []  # All samples
    for i in range(readlim):
        rawframes = f.readframes(sampsize).hex()
        frames = []
        for j in range(len(rawframes)//(channels * sampwidth * 2)):
            hexstring = rawframes[j * channels * sampwidth * 2:(j + 1) * channels * sampwidth * 2]
            currenthex = hexstring[0:sampwidth * 2]
            if(i==0 and j == 1):
                amplitude = decodeAudio(currenthex, 1, 1, 1)
                print(currenthex, amplitude)
            else:
                amplitude = decodeAudio(currenthex, 1, 1, 0)

            if(amplitude > MAX_AMP):
                MAX_AMP = amplitude
            frames.append(amplitude)
        samples.append(frames)
    f.close()
    return samples, framerate, allparams

def process(data, samp_rate):
    freq_data = []
    print(data[0][0])
    for sample in data:
        y = numpy.array(sample)
        yf = np.absolute(rfft(y))
        freq_data.append(yf)

    return freq_data

def deprocess(data):
    new_aud = []
    for sample in data:
        new_aud.append(irfft(sample))
    print(new_aud[0][0])
    return new_aud



def outAudio(mode, fname, params, data):
    # open wav file for writing
    f = wave.open(fname, mode='wb')
    f.setparams(params)
    print(params)
    f.setnframes(0)

    sampwidth = f.getsampwidth()
    f.setnchannels(1)
    # loop through samples
    firstit = 0
    k = 0
    for sample in data:
        for frame in sample:
            binframe = denaryToBinary(frame, 1, sampwidth)
            if(k == 1):
                hexframe = binaryToHex(binframe, 1)
                print(frame)
                print(binframe)
                print(hexframe)
            else:
                hexframe = binaryToHex(binframe, 0)
            k += 1
            f.writeframes(hexframe)
    # decode each sample
    # write decoded sample to file
    # close file
    f.close()

def main():
    # Get input/output filenames from input:
    if not TEST:
        infname = getFname("input file")
        outfname = getFname("output file")
    else:
        infname = "tone.wav"
        outfname = "{0}.wav".format(VNAME)

    # Take in audio as input
    # getAudio (file/live = 0/1, fname)
    print("getting audio")
    t1 = time.time()
    indata, samp_rate_input, params_in = getAudio(0, infname)
    t2 = time.time()
    print("audio taken in: {0}ms".format(t2 - t1))

    x = np.arange(0, len(indata[0]), 1)
    plt.plot(x, indata[0])
    plt.show()
    #print("processing audio")
    #t1 = time.time()
    #infdata = process(indata, samp_rate_input)
    #t2 = time.time()
    #print("done: {0}ms".format(t2 - t1))

    #print("deprocessing audio")
    #t1 = time.time()
    #f_samp_width = params_in[1]
    #outfdata = deprocess(infdata)
    #t2 = time.time()
    #print("done: {0}ms".format(t2-t1))

    #print("outputting audio")
    #t1 = time.time()
    outAudio(0, outfname, params_in, indata)
    #t2 = time.time()
    #print("done: {0}ms".format(t2-t1))
    # Take in audio as comparison output (the 'actual output' to compute loss)
    #outdata, samp_rate_output = getAudio(0, outfname)
    #outfdata = process(outfdata, samp_rate_output)

main()
