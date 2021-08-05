################################################################################

# This is the main file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated:

################################################################################

#                              # IMPORTS #                                 #
from scipy.fft import rfft, rfftfreq
import rnn, wave, numpy, math
from dataconversion import *
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
START_FREQ = 200
STOP_FREQ = 15000
INTERVAL = 100


# The name of the voice (example, for the time being):
VNAME = "test"

################################################################################
def getFname(type):
    #name = input("Please enter the name of {0}: ".format(type))

    # ** Write validation checks here **

    return name

def decodeAudio(hexstring, signed, endianness):
    return binaryToDenary(hexToBinary(hexstring), signed, endianness)

def encodeAudio(frequencies):
    pass

def getAudio(intype, fname):
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
            frame = []
            for c in range(channels):
                currenthex = hexstring[c * sampwidth * 2:(c + 1) * sampwidth * 2]
                amplitude = decodeAudio(currenthex, 1, 0)
                frame.append(amplitude)
            if(frame[0] == 0 and frame[1] == 0):
                print(currenthex)
            frames.append(frame)
        samples.append(frames)
    f.close()

    return samples, framerate, allparams

def process(data, samp_rate):
    freq_data = []
    for sample in data:
        y = numpy.array(sample)
        yf = np.abs(rfft(y))
        freqs = []
        for i in range(START_FREQ, STOP_FREQ, INTERVAL):
            freq_index = int(i * len(sample) * (1/samp_rate))
            freqs.append(yf[freq_index])
        freq_data.append(freqs)

    return freq_data

def deprocess(data, samp_rate):
    sample_data = []
    freq = 0
    print("number of samples: {0}".format(len(data)))
    print("length of first sample: {0}".format(len(data[0])))
    for freqs in data:
        current_sample = np.zeros(samp_rate)
        for i in range(len(freqs)):
            freq = START_FREQ + (i * INTERVAL)
            amplitude = freqs[i]
            x_vals = np.arange(samp_rate).reshape(samp_rate)
            current_sample = current_sample + amplitude * np.sin(x_vals * freq)
        print("done frequency: {0}".format(freq))
        sample_data.append(current_sample)
    return sample_data

def outAudio(mode, fname, params, data):
    # open wav file for writing
    f = wave.open(fname, mode='wb')
    f.setparams(params)
    f.setnframes(0)

    sampwidth = f.getsampwidth()
    channels = f.getnchannels()
    # loop through samples
    firstit = 0
    k = 0
    for sample in data:
        for frame in sample:
            newframe = bytearray()
            for i in range(channels):
                channel_aud = frame[i]
                binframe = denaryToBinary(channel_aud, 0, sampwidth)
                hexframe = binaryToHex(binframe)

                newframe.extend(hexframe)

            f.writeframes(newframe)
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

    # Take in audio as input
    # getAudio (file/live = 0/1, fname)
    print("getting audio")
    t1 = time.time()
    indata, samp_rate_input, params = getAudio(0, infname)
    t2 = time.time()
    print("audio taken in: {0}ms".format(t2 - t1))

    print("processing audio")
    t1 = time.time()
    #infdata = process(indata, samp_rate_input)
    t2 = time.time()
    print("done: {0}ms".format(t2 - t1))

    print("deprocessing audio")
    t1 = time.time()
    #outfdata = deprocess(infdata, samp_rate_input)
    t2 = time.time()
    print("done: {0}ms".format(t2-t1))

    print("outputting audio")
    t1 = time.time()
    outAudio(0, outfname, params, indata)
    t2 = time.time()
    print("done: {0}ms".format(t2-t1))
    # Take in audio as comparison output (the 'actual output' to compute loss)
    #outdata, samp_rate_output = getAudio(0, outfname)
    #outfdata = process(outfdata, samp_rate_output)

main()
