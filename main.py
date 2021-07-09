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
INTERVAL = 50


# The name of the voice (example, for the time being):
VNAME = "borisjohnson"

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

    return samples, framerate

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
    indata, samp_rate_input = getAudio(0, infname)
    infdata = process(indata, samp_rate_input)

    # Take in audio as comparison output (the 'actual output' to compute loss)
    #outdata, samp_rate_output = getAudio(0, outfname)
    #outfdata = process(outfdata, samp_rate_output)

    print(infdata[0])
    print(len(infdata[0]))

main()
