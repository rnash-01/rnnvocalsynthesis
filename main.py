################################################################################

# This is the main file for the Speech-to-Speech synthesis project, for the
# Neural Network training partition of the project.
# Author: Raphael Nash
# Date created: 22/12/2020
# Date updated:

################################################################################

#                              # IMPORTS #                                 #
from scipy.fft import rfft, rfftfreq, irfft
import wave, numpy, math
from dataconversion import *
import matplotlib.pyplot as plt
import numpy as np
import time
from LSTM import LSTM
import os
#from google.colab import files, drive
################################################################################

#                          # GLOBAL CONSTANTS #                            #

TEST = True

# How large the input layer will be/how many frequency bands will be passed
# into it.
INPUT_SIZE = 0

MAX_AMP = 1

# Fourier Offset - ensure that each sample subject to a fourier transform
# transitions smoothly from the previous
# (update to be 1/8 of sample size)
GLOBAL_SAMP_SIZE = int(0.25 * 44100)
merge = GLOBAL_SAMP_SIZE//2
f_offset = GLOBAL_SAMP_SIZE - merge

# The name of the voice (example, for the time being):
VNAME = "test" + str(merge)

################################################################################
def getFname(type):
    name = input("Please enter the name of {0}: ".format(type))

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

    #for j in range(44100):
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
    freq_data = np.array([])

    for i in range((len(data) - samp_size)//f_offset + 2):
        if(samp_size + (i * f_offset) < len(data)):
            sample = data[(i * f_offset):samp_size + (i * f_offset)]
        else:
            sample = data[(i*f_offset):len(data)]

        sample_fft = rfft(sample)
        if(i == 0):
            freq_data = np.abs(sample_fft)
            freq_data = np.reshape(freq_data, (-1, 1))
            print(freq_data)
        else:
            freq_sample = np.reshape(np.abs(sample_fft), (-1, 1))
            if(freq_sample.shape[0] == freq_data.shape[0]):
                freq_data = np.append(freq_data, freq_sample, axis=1)

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


def train(arg_time, arg_learnrate, arg_iterations):
    # Get input/output filenames from input:

    # Take in audio as input
    # getAudio (file/live = 0/1, fname)
    model = ""
    infname = "/content/drive/MyDrive/rnnvocalsynthesis/training/Raph_Nash_1.wav"
    outfname = "/content/drive/MyDrive/rnnvocalsynthesis/training/Boris_Johnson_1.wav"
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

    print("####################################")
    print("Input audio len (frames): {0}".format(len(indata)))
    print("Output audio len (frames): {0}".format(len(newvoicedata)))
    print("####################################")

    print("Processing 'input' audio")
    t1 = time.time()
    infdata = process(indata, samp_size)
    t2 = time.time()
    print("Input audio processed in {0}s".format(t2 - t1))

    print("Processing 'output' audio")
    t1 = time.time()
    newvoicefdata = process(newvoicedata, samp_size)
    t2 = time.time()
    print("Output audio processed in {0}s".format(t2 - t1))

    print("####################################")
    print("Input frequency samples len: {0}".format(len(infdata)))
    print("Output frequency samples len: {0}".format(len(newvoicefdata)))
    print("####################################")

    #print("Normalizing input data (to avoid overflows)")
    #t1 = time.time()

    infdata_mean = np.mean(infdata)
    infdata_std = np.std(infdata)
    infdata = (infdata - infdata_mean)/infdata_std
    #t2 = time.time()
    #print("Done: {0}s".format(t2 - t1))

    #print("Normalizing output data (to avoid overflows)")
    #t1 = time.time()

    newvoicefdata_mean = np.mean(newvoicefdata)
    newvoicefdata_std = np.std(newvoicefdata)
    newvoicefdata = (newvoicefdata - newvoicefdata_mean)/newvoicefdata_std
    #t2 = time.time()
    #print("Done: {0}s".format(t2 - t1))

    print("input vector size: {0}".format(infdata.shape[0]))
    #print("output vector size: {0}".format(newvoicefdata.shape[0]))

    print("Commencing RNN training")
    internals = infdata.shape[0]
    model = LSTM(internals, internals, [internals], [internals], [internals], [42, internals])

    if(os.path.exists("boris_params.txt")):
        model.load_parameters("boris_params.txt")

    t1 = time.time()
    model.train(infdata, newvoicefdata, arg_time, arg_learnrate, arg_iterations, "boris_loss_graph.png")
    t2 = time.time()
    print("Trained in {0}s".format(t2-t1))
    print("Saving parameters...")
    t1 = time.time()
    model.save_parameters("boris_params.txt")
    t2 = time.time()
    print("Saved. Time taken: {0}s".format(t2 - t1))

    print("Creating test output based on training input")
    t1 = time.time()
    test_out = (model.predict(infdata) * (indata_max - indata_min)) + indata_min
    outdata = deprocess(test_out.T)
    outAudio(0, "boris_train_out.wav", params_in, outdata)
    t2 = time.time()
    print("Done. Time taken {0}s".format(t2-t1))

    print("All done! :)")
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

def main():
    print(f_offset)
    samp_size = GLOBAL_SAMP_SIZE
    infname = "tests/input.wav"
    indata, frame_rate, params_in = getAudio(0, infname)
    frequencies = process(indata, samp_size)
    norm = np.linalg.norm(frequencies)
    frequencies = frequencies/norm

    # Create model, load parameters
    in_out_size = frequencies.shape[0]
    model = LSTM(in_out_size, in_out_size)
    model.load_parameters("boris_params.txt")
    newfrequencies = model.predict(frequencies) * norm
    outdata = deprocess(newfrequencies.T)
    outAudio(0, "tests/output.wav", params_in, outdata)

train(10, 0.5, 1000)
