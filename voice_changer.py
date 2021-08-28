from LSTM import LSTM
import dataconversion
from main import process, deprocess, getAudio, outAudio, encodeAudio, decodeAudio

GLOBAL_SAMP_SIZE = int(0.08 * 44100)
merge = GLOBAL_SAMP_SIZE//2
f_offset = GLOBAL_SAMP_SIZE - merge

def main():
    model = LSTM(1765, 1765)
    model.load_parameters("boris_params.txt")

    print("Getting audio:")
    X = getAudio(0, "tests/test.wav")
    frequencies = process(X, GLOBAL_SAMP_SIZE)
    model
main()
