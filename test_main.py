############################# IMPORTS ##########################################

# to test, drag file into test folder so that python can detect it in runtime
from test_functions import *
from main import *

############################ NECESSARY GLOBALS #################################


############################## TESTING BEGINS HERE #############################

def t_getAudio():
    startTest("getAudio")
    print("Test 1")
    it()
    try:
        frames = getAudio(0, "test_audio.wav")
        print(frames)
        print("Got audio with no runtime errors. Please inspect for logic errors.")
        print("Assuming a pass")
        ins()
    except Exception as e:
        print("Failed")
        printerror(e)

    finishtest()
    return score

def t_process():
    startTest("process")
    print("Test 1")
    it()
    try:
        frames = getAudio(0, "tone.wav")
        process(frames)
    except Exception as e:
        print("Failed")
        printerror(e)

def main():
    t_getAudio()
    t_processAudio()


t_process()
