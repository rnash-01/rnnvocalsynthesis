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
from matplotlib import pyplot as plt

################################################################################

#                          # GLOBAL CONSTANTS #                            #

SAMPLE_SIZE = 0  # How large a given audio sample will be. Max size.

# How large the input layer will be/how many frequency bands will be passed
# into it.
INPUT_SIZE = 0

# The name of the voice (example, for the time being):
VNAME = "borisjohnson"

################################################################################
def getFname(type):
    #name = input("Please enter the name of {0}: ".format(type))

    # ** Write validation checks here **

    return name

def decodeAudio(hexstring, signed, endianness):
    return binaryToDenary(hexToBinary(hexstring), signed, endianness)

def encodeAudio():
    pass

def getAudio(intype, fname):
    f = wave.open(fname, 'rb')

    # Get some parameters of the file
    channels = f.getnchannels()
    sampwidth = f.getsampwidth()        # Bytes per sample
    framerate = f.getframerate()        # Frames per second
    n = f.getnframes()                  # Total number of frames

    # Create some of my own parameters
    sampsize = framerates               # Frames per sample
    readlim = math.ceil(n/sampsize)     # Number of samples we'll read

    # Read in all data into samples
    samples = []  # All samples
    for i in range(readlim):
        rawframes = f.readframes(sampsize).hex()
        frames = []
        for j in range(len(rawframes) - 1):
            hexstring = rawframes[j * channels * sampwidth:(j + 1) * channels * sampwidth]
            frame = 0
            for c in range(channels):
                currenthex = hexstring[c * sampwidth:(c + 1) * channels]
                amplitude = decodeAudio(currenthex, 1, 0)
                frame += amplitude/channels
            frames.append(frame)
        samples.append(frames)

    f.close()

    return samples


def process(data):
    all_data = []
    for sample in data:
        x = numpy.array(sample)
        y = np.abs(rfft(x))
        all_data.append(y)

    return all_data

def main():
    # Get input/output filenames from input:
    if not TEST:
        infname = getFname("input file")
        outfname = getFname("output file")
    else:
        infname = "me.wav"
        outfname = "{0}.wav".format(VNAME)

    # Take in audio as input
    # getAudio (file/live = 0/1, fname)
    indata = getAudio(0, SAMPLE_SIZE, infname)

    # Take in audio as comparison output (the 'actual output' to compute loss)
    outdata = getAudio(0, SAMPLE_SIZE, outfname)

    # Process data, and convert into series of arrays containing the
    # values of frequency bands for each sample:
    # [[20Hz, 40Hz, ... 4000Hz], [20Hz, 40Hz, ... 4000Hz], ...]
    # ('x'Hz represents the value of the FFT at 'x'Hz, not simply the frequency
    # itself).
    infdata = process(indata, SAMPLE_SIZE)
    outfdata = process(outfdata, SAMPLE_SIZE)

    # Train Neural Network on frequency data
    converter = rnn.MyRNN()
    converter.train(infdata, outfdata)

    # Save network parameters
    saveFile(converter, "{0}_params".format(VNAME))
