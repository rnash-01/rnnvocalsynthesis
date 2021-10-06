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
from feature_extraction import feature_extract
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

    # First sample to append:
    new_samples = np.concatenate((new_samples, first))
    samp_size = len(new_samples)

    lim = len(data)
    for i in range(1, lim):
        aud_data = irfft(data[i])
        merge = aud_data[0:samp_size - f_offset]

        plot = merge
        x = np.arange(len(plot))

        if(i == lim//2):
            plt.plot(x, plot)
            plt.plot(x, new_samples[(f_offset * i):(f_offset * (i-1) + samp_size)])

        coeff_step = 1/(len(merge) + 1)

        for j in range(len(merge)):
            val = (1 - ((j + 1) * coeff_step)) * new_samples[(f_offset * i) + j] + ((j + 1) * coeff_step * merge[j])
            new_samples[(f_offset * i) + j] = val

        if(i == lim//2):
            plt.plot(x, new_samples[(f_offset * i):(f_offset * (i-1) + samp_size)])
            plt.title("t = " + str((f_offset * i)/44100))
            plt.savefig("latest_plot.png")

        non_merge = aud_data[samp_size - f_offset: samp_size]
        new_samples = np.concatenate((new_samples, non_merge))

        if(i == 1):
            print(new_samples)
    return new_samples

def outAudio(mode, fname, params, data):
    # open wav file for writing
    f = wave.open(fname, mode='wb')
    f.setparams(params)
    f.setnframes(0)

    sampwidth = f.getsampwidth()
    f.setnchannels(1)
    # loop through samples
    firstit = 0
    k = 0
    for frame in data:
        binframe = denaryToBinary(frame, 1, sampwidth)
        if(k == 1):
            hexframe = binaryToHex(binframe, 1)
        else:
            hexframe = binaryToHex(binframe, 0)
        k += 1
        f.writeframes(hexframe)
    # decode each sample
    # write decoded sample to file
    # close file
    f.close()

def mini_batchify(X, mini_batch_size):
    mini_batches = []
    for i in range(math.ceil(X.shape[1] / mini_batch_size)):
        mini_batch = np.empty((X.shape[0], 1))
        if ((i + 1) * mini_batch_size > X.shape[1]):
            mini_batch = X[:,i * mini_batch_size:X.shape[1]]
            remainder = mini_batch_size - X.shape[1] % mini_batch_size
            mini_batch = np.append(mini_batch, np.zeros((X.shape[0], remainder)))
        else:
            mini_batch = X[:, i * mini_batch_size:(i + 1) * mini_batch_size]

        mini_batches.append(mini_batch)

    return mini_batches

def divider(n):
    divide = ""
    for i in range(n):
        divide += "="

    print(divide)

def train(arg_time, arg_learnrate, arg_iterations):
    divider(50)
    version = 0

    # Get input/output filenames from input:
    # Take in audio as input
    infname = "training/Raph_Nash_1.wav"
    newvoicefname = "training/Boris_Johnson_1.wav"
    divider(20)
    print("GET AUDIO")
    divider(20)

    print("Taking in audio from {0}".format(infname))
    indata, samp_rate_input, params_in = getAudio(0, infname)
    print("Done")

    print("Taking in audio from {0}".format(newvoicefname))
    newvoicedata, samp_rate_newvoice, params_newvoice = getAudio(0, newvoicefname)
    print("Done")

    # Process audio
    divider(20)
    print("PROCESS AUDIO")
    divider(20)

    samp_size = GLOBAL_SAMP_SIZE

    print("Processing {0}".format(infname))
    infdata = process(indata, samp_size)
    print("Done")

    print("Processing {0}".format(newvoicefname))
    newvoicefdata = process(newvoicedata, samp_size)
    print("Done")

    # Normalise audio
    divider(20)
    print("NORMALISE AUDIO")
    divider(20)

    print("Normalising input...")
    in_max = np.max(infdata)
    in_min = np.min(infdata)
    infdata = (infdata - in_min)/(in_max - in_min)
    print("Done")
    divider(4)

    print("Normalising new voice data...")
    new_max = np.max(newvoicefdata)
    new_min = np.min(newvoicefdata)
    newvoicefdata = (newvoicefdata - new_min) / (new_max - new_min)
    print("Done")
    divider(4)

    # Initialise networks
    divider(20)
    print("INITIALISE NETWORKS")
    divider(20)

    print("Initialising networks")
    features = 43
    input_autoencoder = feature_extract([infdata.shape[0], features, infdata.shape[0]])
    newvoice_autoencoder = feature_extract([newvoicefdata.shape[0], features, newvoicefdata.shape[0]])

    model = LSTM(features, features, [features], [features], [features], [features])

    print("Done")

    # Train autencoders
    divider(20)
    print("TRAIN AUTOENCODERS")
    divider(20)
    # Set up minibatches
    print("Setting up minibatches")
    mini_batch_size = 10
    mini_batches_input = mini_batchify(infdata, mini_batch_size)
    mini_batches_newvoice = mini_batchify(newvoicefdata, mini_batch_size)
    print("Done")
    print(mini_batches_input[0])
    divider(4)

    # Train
    voice_in_params = "input_params_{0}_{1}.txt".format(features, version)
    voice_out_params = "newvoice_params_{0}_{1}.txt".format(features, version)
    if (os.path.exists(voice_in_params)):
        print("Found {0}. Loading parameters...".format(voice_in_params))
        input_autoencoder.load_parameters(voice_in_params)
        print("Done")
        divider(4)

    print("Training input autoencoder...")
    input_autoencoder.train(mini_batches_input, arg_iterations, 1, "in_voice_cost_{0}.png".format(features))
    print("Done")
    divider(4)

    print("Saving parameters...")
    input_autoencoder.save_parameters(voice_in_params)
    print("Done")
    divider(4)

    if (os.path.exists(voice_out_params)):
        print("Found {0}. Loading parameters...".format(voice_out_params))
        newvoice_autoencoder.load_parameters(voice_out_params)
        print("Done")
        divider(4)

    print("Training output autoencoder...")
    newvoice_autoencoder.train(mini_batches_newvoice, arg_iterations, 1, "out_voice_cost_{0}.png".format(features))
    print("Done")
    divider(4)

    print("Saving parameters...")
    newvoice_autoencoder.save_parameters(voice_out_params)
    print("Done")
    divider(4)

    # Train LSTM
    divider(20)
    print("TRAIN LSTM")
    divider(20)

    print("Creating train input/output")
    LSTM_in = input_autoencoder.inToMiddle(infdata)
    LSTM_out = newvoice_autoencoder.inToMiddle(newvoicefdata)
    print("Done")
    divider(4)

    if(os.path.exists("LSTM_{0}.txt".format(features))):
        print("Found {0}. Loading parameters".format("LSTM_{0}.txt".format(features)))
        model.load_parameters("LSTM_{0}.txt".format(features))
        print("Done")
        divider(4)

    print("Training LSTM...")
    model.train(LSTM_in, LSTM_out, arg_time, arg_learnrate, arg_iterations, "LSTM_cost.png")
    print("Done")
    divider(4)

    print("Saving parameters...")
    model.save_parameters("LSTM_{0}.txt".format(features))
    print("Done")
    divider(4)

    # Create output based on training
    divider(20)
    print("CREATING TEST OUTPUT")
    divider(20)

    print("Passing through neural net")
    LSTM_test_out = model.predict(LSTM_in)
    print("Decoding")
    outfdata = newvoice_autoencoder.middleToOut(LSTM_test_out)
    print("Deprocessing")
    outdata = ((deprocess(outfdata) * (new_max - new_min)) + new_min)/100
    print("Outputting...")
    outAudio(0, "tests/output.wav", params_in, outdata)
    print("DONE! :)")
    divider(50)


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

train(4, 1, 100)
