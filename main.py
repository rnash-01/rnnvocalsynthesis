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

TEST = False

# How large the input layer will be/how many frequency bands will be passed
# into it.
INPUT_SIZE = 0

MAX_AMP = 1

# Fourier Offset - ensure that each sample subject to a fourier transform
# transitions smoothly from the previous
# (update to be 1/8 of sample size)
GLOBAL_SAMP_SIZE = int(0.08 * 44100)
merge = GLOBAL_SAMP_SIZE//2
f_offset = GLOBAL_SAMP_SIZE - merge

# The name of the voice (example, for the time being):
VNAME = "test" + str(merge)

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
    sampsize = int(framerate * 0.1)    # Frames per sample
    readlim = math.ceil(n/sampsize)     # Number of samples we'll read

    # Read in all data into samples
    samples = []  # All samples

    rawframes = f.readframes(n).hex()
    for j in range(len(rawframes)//(channels * sampwidth * 2)):
        hexstring = rawframes[j * channels * sampwidth * 2:(j + 1) * channels * sampwidth * 2]
        currenthex = hexstring[0:sampwidth * 2]
        if(j == 1):
            amplitude = decodeAudio(currenthex, 1, 1, 1)
            print(currenthex, amplitude)
        else:
            amplitude = decodeAudio(currenthex, 1, 1, 0)

        if(amplitude > MAX_AMP):
            MAX_AMP = amplitude
        samples.append(amplitude)
    f.close()
    return samples, framerate, allparams

def process(data, samp_size):
    freq_data = []

    for i in range((len(data) - samp_size)//f_offset + 2):
        if(samp_size + (i * f_offset) < len(data)):
            sample = data[(i * f_offset):samp_size + (i * f_offset)]
        else:
            sample = data[(i*f_offset):len(data)]

        sample_fft = rfft(sample)
        freq_data.append(np.abs(sample_fft))

    return freq_data

def deprocess(data):
    new_samples = []
    first = irfft(data[0])
    print(len(first), len(data[0]))

    # First sample to append:
    new_samples = np.concatenate((new_samples, first))
    samp_size = len(new_samples)
    print(samp_size - f_offset)

    lim = len(data)
    for i in range(1, lim):
        aud_data = irfft(data[i])
        merge = aud_data[0:samp_size - f_offset]

        plot = merge
        x = np.arange(len(plot))

        if(i == lim//2):
            plt.plot(x, plot)
            plt.plot(x, new_samples[(f_offset * i):(f_offset * (i-1) + samp_size)])

        print((f_offset * i)/44100)
        coeff_step = 1/(len(merge) + 1)

        for j in range(len(merge)):
            val = (1 - ((j + 1) * coeff_step)) * new_samples[(f_offset * i) + j] + ((j + 1) * coeff_step * merge[j])
            new_samples[(f_offset * i) + j] = val

        if(i == lim//2):
            plt.plot(x, new_samples[(f_offset * i):(f_offset * (i-1) + samp_size)])
            plt.title("t = " + str((f_offset * i)/44100))
            plt.savefig("latest_plot.png")

        non_merge = aud_data[samp_size - f_offset: samp_size]
        print(non_merge[0]/new_samples[-1], new_samples[-1]/new_samples[-2])
        new_samples = np.concatenate((new_samples, non_merge))

        if(i == 1):
            print(new_samples)
    return new_samples

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
    print(len(data))
    for frame in data:
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
        infname = "test_input.wav"
        outfname = "{0}.wav".format(VNAME)

    model = LSTM(GLOBAL_SAMP_SIZE, GLOBAL_SAMP_SIZE)
    # Take in audio as input
    # getAudio (file/live = 0/1, fname)
    print("Getting comparable input audio")
    t1 = time.time()
    indata, samp_rate_input, params_in = getAudio(0, infname)
    t2 = time.time()
    print("Audio taken in: {0}s".format(t2 - t1))

    # Take in compared output
    print("Getting comparable output audio")
    t1 = time.time()
    newvoicedata, samp_rate_newvoice, params_newvoice = getAudio(0, outfname)
    t2 = time.time()
    print("Audio taken in: {0}s".format(t2 - t1))

    samp_size = GLOBAL_SAMP_SIZE

    print("Processing 'input' audio")
    t1 = time.time()
    infdata = process(indata, samp_size)
    t2 = time.time()
    print("Input audio processed in {0}s".format(t2 - t1))

    print("Processing 'output' audio")
    t1 = time.time()
    newvoicefdata = process(newvoicefdata, samp_size)
    t2 = time.time()
    print("Output audio processed in {0}s".format(t2 - t1))

    print("Commencing RNN training")
    t1 = time.time()
    model.train(infdata, newvoicefdata, 11025, 0.1, 1000)
    t2 = time.time()
    print("Time taken: {0}s".format(t2 - t1))
    print("Done. :)")

    #f_samp_width = params_in[1]
    #outfdata = deprocess(infdata)

    #t2 = time.time()
    #print("done: {0}ms".format(t2-t1))

    #print("outputting audio")
    #t1 = time.time()
    #outAudio(0, outfname, params_in, outfdata)
    #t2 = time.time()
    #print("done: {0}ms".format(t2-t1))
    # Take in audio as comparison output (the 'actual output' to compute loss)
    #outdata, samp_rate_output = getAudio(0, outfname)
    #outfdata = process(outfdata, samp_rate_output)

main()
