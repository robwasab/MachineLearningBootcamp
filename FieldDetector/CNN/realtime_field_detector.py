
# ContinuousMicrophone represents a microphone where we can receive sample data
# Runloop is a class that contains an infinite loop to run enqueued tasks
from microphone import ContinuousMicrophone, Runloop

# utility to list the usb microphones attached to computer
from microphone import list_microphones

# FeatureGenerator takes recorded samples and generates spectograms from them
# RealTimeCNN is a convolutional neural network that runs in realtime using numpy libraries
# MyCNN is a pytorch convolutional neural network. We need to import it so we can load it from file
from realtime import FeatureGenerator, RealTimeCNN, MyCNN, DynamicLayer, StaticLayer
import argparse
import pyaudio
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import time

print('done importing')

def init_args():
    '''
    ArgumentParser
    --list allows us to list attached microphones
    --mics [mic1 [mic2 [mic3]]]
    '''

    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        usage = "Use '%s --help' for more information"%sys.argv[0]
    )

    group = parser.add_mutually_exclusive_group()

    list_help = 'List microphones'
    group.add_argument(
        '--list',
        help=list_help,
        action='store_true',
        default=False)


    mic_help = 'Names of mics to sample from'
    group.add_argument(
        '--mics',
        help=mic_help,
        nargs='*')

    parsed_args = parser.parse_args(args)

    return vars(parsed_args)

# ------------------------------------------------------------------------------


# Initialize argument parser
args = init_args()

# Pyaudio instance for reading from microphones
pa = pyaudio.PyAudio()

if args['list']:
    list_microphones(pa)
    sys.exit(0)

if 0 == len(args['mics']):
    print("must provide a microphone name to use...")
    sys.exit(-1)


# Realtime Convolutional Neural Network
fd = RealTimeCNN()


# Variables for plotting the output of the CNN
history_index = 0
history_length = 128

# history starts out with all zeros
history = np.zeros((history_length, 4))

# Create a new figure
plt.figure()

# Create 4 axes to plot the four outputs of the CNN
history_axes = [
    plt.subplot(4, 1, k+1)
    for k in range(4)
]


# Axis titles
titles = ['Ambient', 'Cold', 'Warm', 'Hot']

# Axis colors
colors = ['b', 'g', 'y', 'r']

# Axis handles
history_handles = []

# Axis vertical line
history_vlines = []

# Initialize plots with zeros
for k, ax in enumerate(history_axes):
    ax.set_ylim((-0.1, 1.1))

    history_handles.append(
        ax.plot(history[:, k], colors[k], linewidth=5
    )[0])

    ax.set_title(titles[k])

    history_vlines.append(
        ax.axvline(x=0, color='r', linewidth=3)
    )

# Display the plots
plt.tight_layout()
plt.show(block=False)


# Func to enqueue prediction in circular history buffer
def push_prediction(p):
    global history_index
    history[history_index, :] = p
    history_index = (history_index + 1) % history_length

# Task replot plot
def task_refresh_plot():
    plt.pause(.001)


# Replot history
def replot_history():
    for k, handle in enumerate(history_handles):
        handle.set_ydata(history[:, k])
        history_vlines[k].set_xdata(history_index)

    main.enqueue_task(task_refresh_plot)


# Task to take features and pass them into CNN
def task_detect_features(features):
    print('detect features')

    tensor = np.array(features)
    ps = fd.predict(tensor)

    # Enqueue predictions into plot
    for p in ps:
        #print(' '.join('%.3f'%x for x in p))
        push_prediction(p)

    replot_history()


# Number of microphones
num_mics = len(args['mics'])


# Microphone sample callback
def task_done_callback(pos, signal):
    global features

    features[pos] = gens[pos].convolve(signal)

    # Only run the detect features task once all microphones have called
    # task_done_callback
    done = True
    for f in features:
        if f is None:
            done = False
            break

    if done:
        # All microphones have generated features
        main.enqueue_task(task_detect_features, features)

        # Reset the features to None
        features = [
            None
            for k in range(num_mics)
        ]


fs = 4000

# Downsample factor to get an effective sample rate of 125 Hz
downsample_factor = 32


# Instantiate microphones
mics = [
    ContinuousMicrophone(
        pa,
        mic_name,
        fs,
        downsample_factor,
        k,
        task_done_callback,
        1*4096, # number of samples to collect before calling task_done_callback
    )
    for k, mic_name in enumerate(args['mics'])
]


# Instantiate feature generators
gens = [
    FeatureGenerator(fs, downsample_factor, bw=5)
    for k in range(len(args['mics']))
]


# Array to hold features once they are calculated
features = [
    None
    for k in range(len(args['mics']))
]

# Mainloop
main = Runloop(debug=True)

# Start recording
for mic in mics:
    mic.start_recording()


while True:
    main.main_loop_task()
    plt.pause(.01)
    for mic in mics:
        mic.main_loop_task()
