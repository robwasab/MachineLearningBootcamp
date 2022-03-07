from microphone import Microphone
from microphone import list_microphones
import argparse
import keyboard
import pyaudio
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from time import strftime
from queue import Queue
import pickle as pkl
import scipy.signal

fs = 4000


def create_dir(output_dir):
    try:
        os.mkdir(output_dir)
        print('created: %s'%output_dir)
    except FileExistsError:
        pass


def init_args():
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        usage = "Use '%s --help' for more information"%sys.argv[0]
    )

    group = parser.add_mutually_exclusive_group()
    label_help = 'Class label to annotate data with'
    group.add_argument(
        '--label',
        help=label_help,
        type=str)

    list_help = 'List microphones'
    group.add_argument(
        '--list',
        help=list_help,
        action='store_true',
        default=False)


    mic_help = 'Names of mics to sample from'
    parser.add_argument(
        '--mics',
        help=mic_help,
        nargs='*')

    pos_help = 'Positions of mics'
    parser.add_argument(
        '--pos',
        help=pos_help,
        nargs='*')

    sec_help = 'Seconds to record'
    parser.add_argument(
        '--secs',
        help=sec_help,
        type=int,
        default=4)

    parsed_args = parser.parse_args(args)

    return vars(parsed_args)


counter = 0

def generate_filename(label):
    global counter
    counter += 1
    return strftime(f'{label}_{counter}_%m%d%Y_%H%M%S.pkl')


lpf = scipy.signal.firwin(256, 10/fs)


def finished_callback(pos, signal):
    global signals
    global label
    global output_dir
    print('finsihed callback: %s with length: %d'%(
            pos,
            len(signal)
        )
    )

    signals[pos] = scipy.signal.convolve(signal, lpf, mode='valid')

    if len(signals) == num_rows:
        # we've gathered all the signals, write it to file
        filename = generate_filename(label)
        filepath = os.path.join(output_dir, filename)
        print('writing ', filepath)
        with open(filepath, 'wb') as f:
            pkl.dump(signals, f)

        print(label)
        if label != 'AMBIENT':
            for pos in signals:
                t = np.arange(len(signals[pos])) / fs
                index = pos2row[pos]
                plt.subplot(num_rows, 1, index)
                plt.plot(t, signals[pos])
                plt.title('%s %d'%(pos, counter) )
                plt.ylim((-0.05, 0.05))
                plt.show(block=False)
                plt.tight_layout()
        else:
            if counter < 50:
                cmd_queue.put(task_start_recording)


args = init_args()

pa = pyaudio.PyAudio()

if args['list']:
    list_microphones(pa)
    sys.exit(0)

if len(args['mics']) != len(args['pos']):
    print('number of mics must equal to positions provided')
    sys.exit(-1)

if 0 == len(args['mics']):
    print("must provide a microphone name to use...")
    sys.exit(-1)


label = args['label']

pos2row = dict((pos, k+1) for k, pos in enumerate(args['pos']))

num_rows = len(args['pos'])

output_dir = 'MultiSensor'
create_dir(output_dir)

cmd_queue = Queue()


mics = [
    Microphone(
        pa,
        mic_name,
        fs,
        pos,
        finished_callback,
        samples2collect=fs*args['secs'],
    )
    for mic_name, pos in zip(args['mics'], args['pos'])
]


def all_not_recording():
    not_recording = True
    for mic in mics:
        not_recording &= not mic.state_record
    return not_recording


def toggle_recording():
    cmd_queue.put(task_start_recording)

signals = None

def task_start_recording():
    global fig
    global signals

    if all_not_recording():
        # starting the recording
        plt.close('all')
        signals = {}
    #else:
    #    # stopping the recording
    #    fig = plt.figure()
    #    fig.set_size_inches(11, 5)

    for mic in mics:
        mic.start_recording()


keyboard.add_hotkey(
    'space',
    toggle_recording,
)


while True:
    if not cmd_queue.empty():
        cmd = cmd_queue.get()
        cmd()

    for mic in mics:
        mic.main_loop_task()

    if all_not_recording():
        plt.pause(.1)
