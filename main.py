################################################################################

# This is the main file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated:

################################################################################

#                              # IMPORTS #                                 #
from scipy.fft import fft
import rnn, wave

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

def decodeAudio():
    pass

def encodeAudio():
    pass

def getAudio(intype, samp_size, fname):
    f = wave.open(fname, 'rb')

    # Get some parameters of the file
    channels = f.getnchannels()
    sampwidth = f.getsampwidth()        # Bytes per sample
    framerate = f.getframerate()        # Frames per second
    n = f.getnframes()                  # Total number of frames

    # Create some of my own parameters
    sampsize = 1024                     # Frames per sample
    readlim = math.ceil(n/sampsize)     # Number of samples we'll read

    samples = []  # All samples
    for i in range(readlim):
        frames = []  # What we'll store our processed frames in
        rawframes = f.readframes(sampsize).hex()
        for j in range(sampsize - 1):
            hexstring = rawframes[j * sampsize:(j + 1) * sampsize]
            amplitude = binaryToDenary(hexToBinary(hexstring), signed, endianness)
            frames.append(amplitude)

    f.close()


def process(data):
    pass

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
